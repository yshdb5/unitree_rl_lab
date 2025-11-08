import torch
from isaaclab_rl.rsl_rl.algorithms.ppo import PPO as BasePPO

def _kl_diag_gauss(mean_p, log_std_p, mean_q, log_std_q):
    var_p, var_q = torch.exp(2*log_std_p), torch.exp(2*log_std_q)
    return 0.5 * ( ((var_p + (mean_p - mean_q)**2) / var_q).sum(-1)
                   - mean_p.size(-1) + 2*(log_std_q - log_std_p).sum(-1) )

class PPO(BasePPO):
    """
    Extends rsl_rl PPO: adds annealed KL(student || teacher).
    Expect rollout["teacher_obs"] and self.teacher to be set.
    """
    def __init__(self, *args, teacher=None, kl_lambda_start=1.0, kl_anneal_updates=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.kl_lambda_start = kl_lambda_start
        self.kl_anneal_updates = kl_anneal_updates
        self._update_counter = 0

    def update(self, rollouts):
        self._update_counter += 1
        losses = super().update(rollouts, return_losses=True)
        if self.teacher is None:
            return losses

        with torch.no_grad():
            t_logp, _, _, t_dist = self.teacher.evaluate_actions(
                rollouts.data["teacher_obs"].view(-1, rollouts.data["teacher_obs"].shape[-1]),
                rollouts.data["actions"].view(-1, rollouts.data["actions"].shape[-1]),
            )
        s_logp, _, _, s_dist = self.actor_critic.evaluate_actions(
            rollouts.data["observations"].view(-1, rollouts.obs_shape[-1]),
            rollouts.data["actions"].view(-1, rollouts.data["actions"].shape[-1]),
        )
        kl = _kl_diag_gauss(s_dist.mean, s_dist.stddev.log(), t_dist.mean, t_dist.stddev.log())  # [N]
        frac = min(1.0, float(self._update_counter) / max(1, self.kl_anneal_updates))
        lam = (1.0 - frac) * self.kl_lambda_start
        self.optimizer.zero_grad()
        total_loss = losses["actor_loss"] + lam * kl.mean()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {**losses, "kl_student_teacher": kl.mean().item(), "kl_lambda": lam}