from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""           
    empirical_normalization = False
    num_envs = 4096
    init_at_random_ep_len = True
    resume = False
    load_run = -1
    load_checkpoint = "latest"

@configclass
class TeacherActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "tasks.locomotion.agents.teacher_actor_critic.MLPActorCritic"
    init_args = {
        "hidden": (512, 512, 256, 128),
    }

@configclass
class TeacherPPOCfg(RslRlPpoAlgorithmCfg):
    class_name = "PPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 3.0e-4
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0

@configclass
class TeacherRunnerCfg(BasePPORunnerCfg):
    actor_critic = TeacherActorCriticCfg()
    algorithm = TeacherPPOCfg()

@configclass
class StudentActorCriticCfg(RslRlPpoActorCriticCfg):
    # causal transformer over L=16 history steps
    class_name = "tasks.locomotion.agents.transformer_actor_critic.TransformerActorCritic"
    init_args = {
        "context_len": 16,
        "d_model": 192,
        "nhead": 4,
        "nlayers": 4,
        "token_mlp": 512,
    }

@configclass
class StudentPPOCfg(RslRlPpoAlgorithmCfg):
    class_name = "KLPPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 3.0e-4
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0

    kl_lambda_start: float = 1.0
    kl_anneal_updates: int = 1000
    teacher_ckpt_path: str = "teacher.pt"

@configclass
class StudentRunnerCfg(BasePPORunnerCfg):
    actor_critic = StudentActorCriticCfg()
    algorithm = StudentPPOCfg()
