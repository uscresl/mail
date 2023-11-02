import contextlib
from typing import Any, Dict, Optional

import numpy as np
import torch
from rlf.algos.base_net_algo import BaseNetAlgo
from rlf.baselines.vec_env import VecEnvWrapper
from rlf.policies.base_policy import get_step_info
from rlf.rl import utils
from rlf.rl.envs import get_vec_normalize, make_vec_envs
from rlf.rl.evaluation import full_eval, train_eval
from rlf.policies.base_policy import get_empty_step_info

import wandb

class Runner:
    """
    Trains a policy
    """

    def __init__(
        self, envs, storage, policy, log, env_interface, checkpointer, args, updater
    ):
        self.envs = envs
        self.storage = storage
        self.policy = policy
        self.log = log
        self.env_interface = env_interface
        self.checkpointer = checkpointer
        self.args = args
        self.updater = updater
        self.train_eval_envs = None

        if self.policy.requires_inference_grads():
            self.train_ctx = contextlib.nullcontext
        else:
            self.train_ctx = torch.no_grad

    def training_iter(self, update_iter: int) -> Dict[str, Any]:
        self.log.start_interval_log()
        self.updater.pre_update(update_iter)

        for step in self.updater.get_steps_generator(update_iter):
            # Sample actions
            obs = self.storage.get_obs(step)

            step_info = get_step_info(update_iter, step, self.episode_count, self.args)

            with self.train_ctx():
                ac_info = self.policy.get_action(
                    utils.get_def_obs(obs, self.args.policy_ob_key),
                    utils.get_other_obs(obs),
                    self.storage.get_hidden_state(step),
                    self.storage.get_masks(step),
                    step_info,
                )
                if self.args.clip_actions:
                    ac_info.clip_action(*self.ac_tensor)

            next_obs, reward, done, infos = self.envs.step(ac_info.take_action)

            reward += ac_info.add_reward

            step_log_vals = utils.agg_ep_log_stats(infos, ac_info.extra)

            self.episode_count += sum([int(d) for d in done])
            self.log.collect_step_info(step_log_vals)

            self.storage.insert(obs, next_obs, reward, done, infos, ac_info)
        updater_log_vals = self.updater.update(self.storage)

        self.storage.after_update()

        return updater_log_vals

    @property
    def should_start_with_eval(self) -> bool:
        """
        If true, will evaluate the policy before the main training loop begins.
        """
        return False

    def setup(self) -> None:
        """
        Runs before any evaluation or training.
        """
        self.episode_count = 0
        self.alg_env_settings = self.updater.get_env_settings(self.args)
        self.updater.first_train(self.log, self._eval_policy, self.env_interface)
        if self.args.clip_actions:
            self.ac_tensor = utils.ac_space_to_tensor(self.policy.action_space)

    def _eval_policy(self, policy, total_num_steps, args) -> Optional[VecEnvWrapper]:
        return train_eval(
            self.envs,
            self.alg_env_settings,
            policy,
            args,
            self.log,
            total_num_steps,
            self.env_interface,
            self.train_eval_envs,
        )

    def log_vals(self, updater_log_vals, update_iter):
        total_num_steps = self.updater.get_completed_update_steps(update_iter + 1)
        return self.log.interval_log(
            update_iter,
            total_num_steps,
            self.episode_count,
            updater_log_vals,
            self.args,
        )

    def save(self, update_iter: int) -> None:
        if (
            (self.episode_count > 0) or (self.args.num_steps == 0)
        ) and self.checkpointer.should_save():
            vec_norm = get_vec_normalize(self.envs)
            if vec_norm is not None:
                self.checkpointer.save_key("ob_rms", vec_norm.ob_rms_dict)
            self.checkpointer.save_key("step", update_iter)

            self.policy.save_to_checkpoint(self.checkpointer)
            self.updater.save(self.checkpointer)

            self.checkpointer.flush(num_updates=update_iter)
            if self.args.sync:
                self.log.backup(self.args, update_iter + 1)

    def eval(self, update_iter):
        if (
            (self.episode_count > 0)
            or (self.args.num_steps <= 1)
            or self.should_start_with_eval
        ):
            total_num_steps = self.updater.get_completed_update_steps(update_iter + 1)
            self.train_eval_envs = self._eval_policy(
                self.policy, total_num_steps, self.args
            )

    def softgym_eval(self, update_iter):
        total_num_steps = self.updater.get_completed_update_steps(update_iter + 1)
        print(f'Evaluation at {total_num_steps} num_steps')

        # evaluate policy at this timestep
        self.policy.eval()
        hidden_states = {}
        for k, dim in self.policy.get_storage_hidden_states().items():
            hidden_states[k] = torch.zeros(1, dim).to(self.args.device)
        eval_masks = torch.zeros(1, 1, device=self.args.device)

        num_eval_episodes = 10
        total_rewards, total_lengths, total_normalized_perf = 0, 0, []
        for j in range(num_eval_episodes):
            o, d, ep_ret, ep_len = self.envs.venv.envs[0].reset(is_eval=True), False, 0, 0
            ep_normalized_perf = []
            while not d:
                # get action
                step_info = get_empty_step_info()
                with torch.no_grad():
                    o = o[None, :, :, :].to(self.args.device)
                    ac_info = self.policy.get_action(
                        o,
                        {},
                        hidden_states,
                        eval_masks,
                        step_info,
                    )
                    hidden_states = ac_info.hxs

                o, r, d, info = self.envs.venv.envs[0].step(ac_info.take_action)

                # update eval_masks
                eval_masks = torch.tensor([[0.0] if d else [1.0]], dtype=torch.float32, device=self.args.device)

                ep_ret += r
                ep_len += 1
                ep_normalized_perf.append(info['normalized_performance'])

            total_rewards += ep_ret
            total_lengths += ep_len
            total_normalized_perf.append(ep_normalized_perf)

        avg_rewards = total_rewards / num_eval_episodes
        avg_ep_length = total_lengths / num_eval_episodes
        avg_normalized_perf = np.mean(total_normalized_perf)
        final_normalized_perf = np.mean(np.array(total_normalized_perf)[:, -1])
        if wandb.run:
            wandb.log({
                "val/info_normalized_performance_mean": avg_normalized_perf,
                'val/info_normalized_performance_final': final_normalized_perf,
                "val/avg_rews": avg_rewards,
                "val/avg_ep_length": avg_ep_length,
                "num_timesteps": total_num_steps,
            })

        # save policy's checkpoint at this timestep (copied from the save function)
        vec_norm = get_vec_normalize(self.envs)
        if vec_norm is not None:
            self.checkpointer.save_key("ob_rms", vec_norm.ob_rms_dict)
        self.checkpointer.save_key("step", update_iter)

        self.policy.save_to_checkpoint(self.checkpointer)
        self.updater.save(self.checkpointer)

        self.checkpointer.flush(num_updates=total_num_steps)
        if self.args.sync:
            self.log.backup(self.args, update_iter + 1)

        # Switch policy back to train mode
        self.policy.train()
        print('Finished evaluation!')

    def softgym_full_eval(self):
        from sb3.utils import set_seed_everywhere, make_dir
        import os

        env = self.envs.venv.envs[0]
        checkpoint_folder = "/".join(self.args.load_file.split('/')[:-1])
        eval_video_path = make_dir(checkpoint_folder + '/eval_video')

        # prepare policy for evaluation
        self.policy.eval()
        hidden_states = {}
        for k, dim in self.policy.get_storage_hidden_states().items():
            hidden_states[k] = torch.zeros(1, dim).to(self.args.device)
        eval_masks = torch.zeros(1, 1, device=self.args.device)

        # start evaluation
        total_normalized_perf_final = []
        random_seeds = [100, 201, 302, 403, 504]
        for curr_seed in random_seeds:
            set_seed_everywhere(curr_seed)
            for ep in range(20):
                o, d, ep_rew, ep_len = env.reset(is_eval=True), False, 0, 0
                ep_normalized_perf = []
                env.start_record()
                while not d:
                    # get action
                    step_info = get_empty_step_info()
                    with torch.no_grad():
                        o = o[None, :, :, :].to(self.args.device)
                        ac_info = self.policy.get_action(
                            o,
                            {},
                            hidden_states,
                            eval_masks,
                            step_info,
                        )
                        hidden_states = ac_info.hxs

                    o, r, d, info = env.step(ac_info.take_action)

                    # update eval_masks
                    eval_masks = torch.tensor([[0.0] if d else [1.0]], dtype=torch.float32, device=self.args.device)

                    ep_rew += r
                    ep_len += 1
                    ep_normalized_perf.append(info['normalized_performance'])
                print(f'Seed {curr_seed} Ep {ep} Current Episode Rewards: {ep_rew}, Episode normalized performance final: {ep_normalized_perf[-1]}, Episode Length: {ep_len}, Done: {d}')
                total_normalized_perf_final.append(ep_normalized_perf)
                env.end_record(video_path=os.path.join(eval_video_path, f'ep_{ep}_{ep_normalized_perf[-1]}_picknplace.gif'))

        total_normalized_perf_final = np.array(total_normalized_perf_final)
        npy_file_path = checkpoint_folder + '/eval_five_seeds_results.npy'
        np.save(npy_file_path, total_normalized_perf_final)
        print('!!!!!!! info_normalized_performance_final !!!!!!!')
        print(f'Mean: {np.mean(total_normalized_perf_final):.4f}')
        print(f'Std: {np.std(total_normalized_perf_final):.4f}')
        print(f'Median: {np.median(total_normalized_perf_final):.4f}')
        print(f'25th Percentile: {np.percentile(total_normalized_perf_final, 25):.4f}')
        print(f'75th Percentile: {np.percentile(total_normalized_perf_final, 75):.4f}')

    def close(self):
        self.log.close()
        if self.train_eval_envs is not None:
            self.train_eval_envs.close()
        self.envs.close()

    def resume(self):
        self.updater.load_resume(self.checkpointer)
        self.policy.load_resume(self.checkpointer)
        return self.checkpointer.get_key("step")

    def should_load_from_checkpoint(self):
        return self.checkpointer.should_load()

    def full_eval(self, create_traj_saver_fn):
        alg_env_settings = self.updater.get_env_settings(self.args)

        tmp_env = make_vec_envs(
            self.args.env_name,
            self.args.seed,
            1,
            self.args.gamma,
            self.args.device,
            False,
            self.env_interface,
            self.args,
            alg_env_settings,
            set_eval=False,
        )
        vec_norm = None
        if self.checkpointer.has_load_key("ob_rms"):
            ob_rms_dict = self.checkpointer.get_key("ob_rms")
            vec_norm = get_vec_normalize(tmp_env)
            if vec_norm is not None:
                vec_norm.ob_rms_dict = ob_rms_dict

        return full_eval(
            self.envs,
            self.policy,
            self.log,
            self.checkpointer,
            self.env_interface,
            self.args,
            alg_env_settings,
            create_traj_saver_fn,
            vec_norm,
        )

    def load_from_checkpoint(self):
        self.policy.load_state_dict(self.checkpointer.get_key("policy"))

        if self.checkpointer.has_load_key("ob_rms"):
            ob_rms_dict = self.checkpointer.get_key("ob_rms")
            vec_norm = get_vec_normalize(self.envs)
            if vec_norm is not None:
                vec_norm.ob_rms_dict = ob_rms_dict
        self.updater.load(self.checkpointer)
