import argparse
from tqdm import tqdm
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools 
# import envs.wrappers as wrappers

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()




class Dreamer(nn.Module):
    def __init__(self,
                 obs_space,
                 act_space,
                 config,
                 buffer_loader,
                 logger,
                 offline_data,
                 online_data,
                 video_recorder,
                 init_meta=False):
        super(Dreamer, self).__init__()

        # reset config
        config.steps //= config.action_repeat
        config.eval_every //= config.action_repeat
        config.log_every //= config.action_repeat
        config.time_limit //= config.action_repeat
        acts = act_space
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

        self._config = config
        # self._logger = logger
        self._should_log = tools.Every(config.log_every)
        self._should_log_policy = tools.Every(1)
        self._should_log_wm = tools.Every(config.log_pretrain_every)
        batch_steps = config.batch_size * config.batch_length
        # TODO:  Edit to run 1/4 of the times then the original dreamer as it realtes to number of parallel runs
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._buffer_loader = buffer_loader
        self._step = buffer_loader.dataset._storage._num_transitions
        self._logger = tools.Logger(logger._log_dir / "tb/", config.action_repeat * self._step, logger.use_wandb)
        self._ulogger = logger
        self._video_recorder = video_recorder

        # self._step = count_steps(config.traindir)
        self._update_count = 0
        self._off_dataset = offline_data
        self._on_dataset = online_data
        self._initial_meta_ready = init_meta

        # Schedules.
        config._set_flag("allow_objects", True)
        config.actor_entropy = lambda x=config.actor_entropy: tools.schedule(
            x, self._step
        )
        config.actor_state_entropy = (
            lambda x=config.actor_state_entropy: tools.schedule(x, self._step)
        )
        config.imag_gradient_mix = lambda x=config.imag_gradient_mix: tools.schedule(
            x, self._step
        )
        # self._dataset = _iterate_episodes(buffer_loader.dataset)
        if type(obs_space) is not dict:
            # obs_space = {obs_space.name: obs_space}
            obs_space = {'image': obs_space}
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(
            config, self._wm, config.behavior_stop_grad
        )
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)


    def init_meta(self):
        if self._initial_meta_ready:
            return {}
        print("Phase I: Pretrain Worls model on offline data")
        self._initial_meta_ready = True
        step = self._step
        steps = (
            self._config.pretrain
            if self._should_pretrain()
            else self._should_train(step)
        )
        print("loading episodes")
        itr_dataset = iter(self._off_dataset)
        for s in tqdm([i for i in range(steps)]):
            self._train(online_data=None, 
                        offline_data=next(itr_dataset),
                        offline=True)
            if s == 0:
                print("finished loading")
            self._update_count += 1
            self._metrics["update_count"] = self._update_count
            if self._should_log_wm(s):
                self.log_metrics(next(itr_dataset), s, sub_prefix="wm")
        return {}

    # def regress_meta(self, replay, step):
    #     return {}

    def update_meta(self, meta, reset, state=None, training=True):
        return meta

    def __call__(self, obs, reset, state=None, training=True):
        return self.forward(obs, reset, state, training)    

    def act(self, obs, meta={}, step=None, eval_mode=False):
        meta = meta['extra_meta']
        timestep = meta["reset"]
        if 'state' in meta:
            state = (meta['state']['latent'], meta['state']['action'])
        else: 
            state = None
        reset = np.array([t.last() for t in [timestep]])
        reward = np.array([meta["reward"]])
        obs = {"image": np.expand_dims(obs, (0)), # expecting b t c h w
               "reward": np.expand_dims(reward, (0)),
               "is_first": np.expand_dims(timestep.first(), (0)),
               "is_last": np.expand_dims(timestep.last(), (0))
               }
        policy, state = self.forward(obs, reset, state, training=not eval_mode)
        state = {"latent": state[0], "action": state[1]}
        meta['extra_meta'] = meta
        meta['extra_meta']['state'] = state
        return policy, meta

    def update(self, replay_iter, global_step):
        steps = (
            self._config.pretrain
            if self._should_pretrain()
            else self._should_train(global_step)
        )
        steps = self._config.reward_finetune_steps if self._config.reward_finetune_steps > 0 else steps
        on_iter = iter(self._on_dataset)
        off_iter = iter(self._off_dataset)
        for _ in range(steps):
            self._train(next(on_iter), next(off_iter), offline=False)
            self._update_count += 1
            self._metrics["update_count"] = self._update_count
        if steps > 0 and self._should_log_policy(global_step):
            self.log_metrics(next(on_iter), global_step)
        return {}

    def log_metrics(self, video_data, step=1, sub_prefix=""):
        with self._ulogger.log_and_dump_ctx(step, ty='train') as log:
            for name, values in self._metrics.items():
                log(f"train_{sub_prefix}_{name}", float(np.mean(values)))
        for name, values in self._metrics.items():
            self._logger.scalar(name, float(np.mean(values)))
            self._metrics[name] = []
        if self._config.video_pred_log:
            openl = self._wm.video_pred(video_data)
            self._logger.video("train_openl", to_np(openl))
        self._logger.write(fps=True)

    @property
    def update_task_every_step(self):
        # return self._config.update_task_every_step
        return 1

    def forward(self, obs, reset, state=None, training=True):
        step = self._step
        if self._should_reset(step):
            state = None
        if state is not None and reset.any():
            mask = 1 - reset
            for key in state[0].keys():
                for i in range(state[0][key].shape[0]):
                    state[0][key][i] *= mask[i]
            for i in range(len(state[1])):
                state[1][i] *= mask[i]

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state
    

    def _policy(self, obs, state, training):
        if state is None:
            batch_size = len(obs["image"])
            latent = self._wm.dynamics.initial(len(obs["image"]))
            action = torch.zeros((batch_size, self._config.num_actions)).to(
                self._config.device
            )
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs) # input obs should have (b,t,3,64,64) and embed should have (b,t,4096) is_first (b,t)
        latent, _ = self._wm.dynamics.obs_step(
            latent, action, embed, obs["is_first"], self._config.collect_dyn_sample
        )
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor_dist == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        action = self._exploration(action, training)
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _exploration(self, action, training):
        amount = self._config.expl_amount if training else self._config.eval_noise
        if amount == 0:
            return action
        if "onehot" in self._config.actor_dist:
            probs = amount / self._config.num_actions + (1 - amount) * action
            return tools.OneHotDist(probs=probs).sample()
        else:
            return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)
        raise NotImplementedError(self._config.action_noise)

    def _select_datasource(self, online_data, offline_data, metrics):
        return offline_data

    def _train(self, online_data, offline_data, offline):
        if offline:
            wm_data = offline_data
        else :
            wm_data = online_data
        metrics = {}
        post, context, mets = self._wm._train(wm_data, offline)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        if not offline:
            data_to_policy = self._select_datasource(online_data, offline_data, mets)
            start, context = self._wm.encode_online(data_to_policy)
            if not self._config.no_task:
                metrics.update(self._task_behavior._train(start, reward)[-1])
            if self._config.expl_behavior != "greedy":
                mets = self._expl_behavior._train(start, context, data_to_policy)[-1]
                metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset_urlb(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size, config.batch_length)
    return dataset

def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, logger, mode, train_eps, eval_eps):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(task, config.action_repeat, config.size)
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task, mode if "train" in mode else "test", config.action_repeat
        )
        env = wrappers.OneHotAction(env)
    elif suite == "MemoryMaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    if (mode == "train") or (mode == "eval"):
        callbacks = [
            functools.partial(
                ProcessEpisodeWrap.process_episode,
                config,
                logger,
                mode,
                train_eps,
                eval_eps,
            )
        ]
        env = wrappers.CollectDataset(env, mode, train_eps, callbacks=callbacks)
    env = wrappers.RewardObs(env)
    return env


class ProcessEpisodeWrap:
    eval_scores = []
    eval_lengths = []
    last_step_at_eval = -1
    eval_done = False

    @classmethod
    def process_episode(cls, config, logger, mode, train_eps, eval_eps, episode):
        directory = dict(train=config.traindir, eval=config.evaldir)[mode]
        cache = dict(train=train_eps, eval=eval_eps)[mode]
        # this saved episodes is given as train_eps or eval_eps from next call
        filename = tools.save_episodes(directory, [episode])[0]
        length = len(episode["reward"]) - 1
        score = float(episode["reward"].astype(np.float64).sum())
        video = episode["image"]
        # add new episode
        cache[str(filename)] = episode
        if mode == "train":
            step_in_dataset = 0
            for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
                if (
                    not config.dataset_size
                    or step_in_dataset + (len(ep["reward"]) - 1) <= config.dataset_size
                ):
                    step_in_dataset += len(ep["reward"]) - 1
                else:
                    del cache[key]
            logger.scalar("dataset_size", step_in_dataset)
        elif mode == "eval":
            # keep only last item for saving memory
            while len(cache) > 1:
                # FIFO
                cache.popitem()
            # start counting scores for evaluation
            if cls.last_step_at_eval != logger.step:
                cls.eval_scores = []
                cls.eval_lengths = []
                cls.eval_done = False
                cls.last_step_at_eval = logger.step

            cls.eval_scores.append(score)
            cls.eval_lengths.append(length)
            # ignore if number of eval episodes exceeds eval_episode_num
            if len(cls.eval_scores) < config.eval_episode_num or cls.eval_done:
                return
            score = sum(cls.eval_scores) / len(cls.eval_scores)
            length = sum(cls.eval_lengths) / len(cls.eval_lengths)
            episode_num = len(cls.eval_scores)
            logger.video(f"{mode}_policy", video[None])
            cls.eval_done = True

        print(f"{mode.title()} episode has {length} steps and return {score:.1f}.")
        logger.scalar(f"{mode}_return", score)
        logger.scalar(f"{mode}_length", length)
        logger.scalar(
            f"{mode}_episodes", len(cache) if mode == "train" else episode_num
        )
        logger.write(step=logger.step)


def main(config):
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step, False)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps)
    train_envs = [make("train") for _ in range(config.envs)]
    eval_envs = [make("eval") for _ in range(config.envs)]
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s, r):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        tools.simulate(random_agent, train_envs, prefill)
        logger.step = config.action_repeat * count_steps(config.traindir)

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest_model.pt").exists():
        agent.load_state_dict(torch.load(logdir / "latest_model.pt"))
        agent._should_pretrain._once = False

    state = None
    while agent._step < config.steps:
        logger.write()
        print("Start evaluation.")
        eval_policy = functools.partial(agent, training=False)
        tools.simulate(eval_policy, eval_envs, episodes=config.eval_episode_num)
        if config.video_pred_log:
            video_pred = agent._wm.video_pred(next(eval_dataset))
            logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        state = tools.simulate(agent, train_envs, config.eval_every, state=state)
        torch.save(agent.state_dict(), logdir / "latest_model.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass

def mainmain():
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--configs", nargs="+")
        args, remaining = parser.parse_known_args()
        configs = yaml.safe_load(
            (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
        )

        def recursive_update(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base:
                    recursive_update(base[key], value)
                else:
                    base[key] = value

        name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
        defaults = {}
        for name in name_list:
            recursive_update(defaults, configs[name])
        parser = argparse.ArgumentParser()
        for key, value in sorted(defaults.items(), key=lambda x: x[0]):
            arg_type = tools.args_type(value)
            parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
        main(parser.parse_args(remaining))
