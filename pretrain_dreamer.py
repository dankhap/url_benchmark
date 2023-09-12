import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
import wandb
import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
from pathlib import Path
from time import sleep
import traceback
import warnings

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

# os.environ['MUJOCO_GL'] = 'osmesa'
# os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
# os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'


from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_orig_replay_loader, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

from dreamer.dreamer import Dreamer
from dreamer.dreamer import make_dataset_urlb

from dmc_benchmark import PRIMAL_TASKS

torch.backends.cudnn.benchmark = True

def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        if "SLURM_JOB_ID" in os.environ:
            print(f'slurm job id: {os.environ["SLURM_JOB_ID"]}')

        self.buffer_dir = self.work_dir
        if cfg.buffer_dir != "":
            self.buffer_dir = Path(cfg.buffer_dir)
            # create the directory if it doesn't exist
            self.buffer_dir.mkdir(parents=True, exist_ok=True)


        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        if cfg.use_wandb:
            full_config = OmegaConf.to_container(cfg, resolve=True)
            exp_name = '_'.join([cfg.experiment,
                cfg.agent.name,
                cfg.domain,
                cfg.obs_type,
                str(cfg.seed),
                "pretrain"])
            wandb.init(project="dreamerv3_urlb",
                entity="urlb-gqn-test",
                group=cfg.group_name,
                name=exp_name,
                config=full_config)
                # sync_tensorboard=True,
                # config=full_config)

        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs
        task = PRIMAL_TASKS[self.cfg.domain]
        self.train_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed, resize=(64, 64))
        self.eval_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                 cfg.action_repeat, cfg.seed, resize=(64, 64))

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        # create replay buffer
        self.replay_loader = make_orig_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                cfg.save_buffer,
                                                cfg.nstep,
                                                cfg.discount,
                                                cfg.seed)
        self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        print("environment initialization complete")

        if "dreamer_conf" in cfg:
            assert cfg.device == cfg.dreamer_conf.dreamer.device
            self.replay_offline_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                      self.buffer_dir / 'buffer')
            # create replay buffer
            self.replay_offline_loader = make_orig_replay_loader(self.replay_offline_storage,
                                                    cfg.replay_buffer_size,
                                                    cfg.batch_size,
                                                    cfg.replay_buffer_num_workers,
                                                    cfg.save_buffer, cfg.nstep, cfg.discount)
            self.dream_online_dataset = make_dataset_urlb(self.replay_loader, cfg.dreamer_conf.dreamer)
            self.dream_offline_dataset = make_dataset_urlb(self.replay_offline_loader, cfg.dreamer_conf.dreamer)
            obs_spec = self.train_env.observation_spec()
            self.agent = Dreamer(
                obs_spec,
                self.train_env.action_spec(),
                cfg.dreamer_conf.dreamer,
                self.replay_loader,
                self.logger,
                self.dream_offline_dataset,
                self.dream_online_dataset,
                self.video_recorder,
                init_meta=True, # true if we skip init_meta pretraining based on external buffer
                offline_loader=self.replay_offline_loader
            ).to(self.device)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def act_warpper(self, time_step, meta, eval_mode):
        meta.update({'extra_meta': {
                        "reset": time_step.step_type,
                        "reward": time_step.reward,
                        }})
        action = self.agent.act(time_step.observation,
                                meta,
                                self.global_step,
                                eval_mode=eval_mode)
        if type(action) is tuple and len(action) == 2:
            # handle output from dreamer
            action, meta = action
            action = action['action'].squeeze(0).cpu().numpy()
            meta = {'extra_meta': meta}
            # should be a numpy array of shape (6,)
        return action, meta

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action, meta = self.act_warpper(time_step, meta, eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        meta = self.agent.init_meta()
        self.replay_storage.add(time_step, meta)
        self.train_video_recorder.init(time_step.observation)
        print("video recorder initialized")
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                meta = self.agent.init_meta()
                self.replay_storage.add(time_step, meta)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                episode_step = 0
                episode_reward = 0

            # Allow taking snapshots mid-episode
            if self.global_frame in self.cfg.snapshots:
                self.save_snapshot()

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            meta = self.agent.update_meta(meta, self.global_step, time_step)
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action, meta = self.act_warpper(time_step, meta, eval_mode=True)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        if type(payload['agent']) == Dreamer:
            payload['agent'] = payload['agent'].state_dict()
            payload['dreamer'] = 1
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        domain, _ = self.cfg.task.split('_', 1)
        # snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.agent.name
        snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.pretrained_agent
        # snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / "icm"

        def try_load(seed):
            snapshot = snapshot_dir / str(
                seed) / f'snapshot_{self.cfg.snapshot_ts}.pt'
            print(f"loading snapshot {snapshot}")
            if not snapshot.exists():
                return None
            with snapshot.open('rb') as f:
                payload = torch.load(f)
            if 'dreamer' in payload:
                payload['agent'] = Dreamer.from_state_dict(payload['agent'])
            return payload

        # try to load current seed
        payload = try_load(self.cfg.seed)
        if payload is not None:
            return payload
        print(f"failed loading snapshot at: {snapshot_dir}")
        # otherwise try random seed
        attempts = 10
        while True:
            attempts -= 1
            sleep(1)
            seed = np.random.randint(1, 11)
            payload = try_load(seed)
            if payload is not None:
                return payload
            if not attempts:
                break
        return None


@hydra.main(config_path='/code/url_benchmark', config_name='pretrain_dreamer')
def main(cfg):
    from pretrain_dreamer import Workspace as W
    os.environ["MUJOCO_GL"] = "egl"
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()

