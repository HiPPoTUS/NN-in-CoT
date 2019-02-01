import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
import pybullet_envs
from datetime import datetime, time

from baselines.common import tf_util as U
from baselines import logger
import gym

dirname = datetime.now().strftime("%y%m%d_%I%M%S")

ENV = 'HumanoidBulletEnv-v0'
NUM = 10000
SEED = 0

def train(env_id, num, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = gym.make(env_id)
    env.seed(seed)
    env.render('human')
    pposgd_simple.learn(env, policy_fn,
            max_iters=num,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='constant',
        )
    env.close()

def main():
	logger.configure(dir='./LOG/{}'.format(dirname),format_strs=['tensorboard','stdout'])
	train(ENV,NUM,SEED)

if __name__=='__main__':
	main()