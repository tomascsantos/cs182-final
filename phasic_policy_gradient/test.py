import argparse
from mpi4py import MPI
from . import ppg
from . import torch_util as tu
from .impala_cnn import ImpalaEncoder
from . import logger
from .envs import get_venv
import torch
from .roller import Roller

def train_fn(env_name="fruitbot",
    distribution_mode="easy",
    arch="dual",  # 'shared', 'detach', or 'dual'
    # 'shared' = shared policy and value networks
    # 'dual' = separate policy and value networks
    # 'detach' = shared policy and value networks, but with the value function gradient detached during the policy phase to avoid interference
    interacts_total=10_000_000,
    num_envs=1,
    n_epoch_pi=1,
    n_epoch_vf=1,
    gamma=.999,
    aux_lr=5e-4,
    lr=5e-4,
    nminibatch=1,
    aux_mbsize=4,
    clip_param=.2,
    kl_penalty=0.0,
    n_aux_epochs=6,
    n_pi=32,
    beta_clone=1.0,
    vf_true_weight=1.0,
    model_dir='results/tmp/ppg_16_10_gray',
    start_level=0,
    comm=None):
    if comm is None:
        comm = MPI.COMM_WORLD
    tu.setup_dist(comm=comm)
    tu.register_distributions_for_tree_util()

    print("num envs: ", num_envs)

    venv = get_venv(num_envs=num_envs, env_name=env_name, distribution_mode=distribution_mode, num_levels=10, start_level=start_level)
    print(venv, env_name, distribution_mode)
    model = torch.load(model_dir + "/model.jd")

    model.to(tu.dev())
    tu.sync_params(model.parameters())
    model.eval()
    test_roller = Roller(
          act_fn=model.act,
          venv=venv,
          initial_state=model.initial_state(venv.num),
          keep_buf=100,
          keep_non_rolling=None,
          p_gray=0.0
      )
    out = test_roller.multi_step(4000)
    print(test_roller.episode_count)
    print(test_roller.recent_eplens, len(test_roller.recent_eplens))
    print(test_roller.recent_eprets)
    print(sum(test_roller.recent_eprets)/100)
      

def main():
    parser = argparse.ArgumentParser(description='Process PPG training arguments.')
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--num_envs', type=int, default=16)
    parser.add_argument('--n_epoch_pi', type=int, default=1)
    parser.add_argument('--n_epoch_vf', type=int, default=1)
    parser.add_argument('--n_aux_epochs', type=int, default=6)
    parser.add_argument('--n_pi', type=int, default=32)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--kl_penalty', type=float, default=0.0)
    parser.add_argument('--arch', type=str, default='dual') # 'shared', 'detach', or 'dual'
    parser.add_argument('--model', type=str)
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD

    train_fn(
        env_name=args.env_name,
        num_envs=args.num_envs,
        n_epoch_pi=args.n_epoch_pi,
        n_epoch_vf=args.n_epoch_vf,
        n_aux_epochs=args.n_aux_epochs,
        n_pi=args.n_pi,
        arch=args.arch,
        model_dir = args.model,
        start_level=args.start,
        comm=comm)

if __name__ == '__main__':
    main()
