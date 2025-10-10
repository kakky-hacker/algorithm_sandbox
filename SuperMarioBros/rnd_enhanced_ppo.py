from my_env import MyMarioEnv
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
from chainerrl.agents.mario import MARIO
from chainerrl.agents.a3c import A3CModel
from chainerrl.misc.batch_states import batch_states
from chainerrl import links
from chainerrl import misc
from chainerrl import policies
import numpy as np
import cupy as cp
from rnd import RND

#pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
#cp.cuda.set_allocator(pool.malloc)

class A3CSoftmax(chainer.Chain, chainerrl.agents.a3c.A3CModel):
    
    def __init__(self):
        super(A3CSoftmax, self).__init__()        
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 4, (15,16), 1, 0)  
            self.conv2 = L.Convolution2D(4, 16, (15,16), 1, 0) 
            self.conv3 = L.Convolution2D(16, 32, (15,16), 1, 0) 
            self.l4p = L.Linear(288, 288)
            self.l4v = L.Linear(288, 288)
            self.l5p = L.Linear(288, 288)
            self.l5v = L.Linear(288, 288)
            self.pi = chainerrl.policies.SoftmaxPolicy(L.Linear(288, 7)) 
            self.v = L.Linear(288, 1) 

    def pi_and_v(self, state):
        h1 = F.max_pooling_2d(F.relu(self.conv1(state)), ksize=2, stride=2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2) 
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), ksize=2, stride=2) 
        h4p = F.relu(self.l4p(h3)) 
        h4v = F.relu(self.l4v(h3))
        h5p = F.relu(self.l5p(h4p)) 
        h5v = F.relu(self.l5v(h4v))
        pout = self.pi(h5p) 
        vout = self.v(h5v) 
        return pout, vout

def main():

    env = MyMarioEnv()
    
    num_episodes = 10**6
    
    model = A3CSoftmax()
    
    optimizer = chainer.optimizers.Adam(alpha=3e-4)
    optimizer.setup(model)
    
    phi = lambda x: x.astype(np.float32, copy=False)
    
    rnd = RND(batchsize=32, alpha=1e-3, epoch=3, gpu=0, reward_coef=1.0, padding=0.0)

    agent = MARIO(model,
                optimizer,
                obs_normalizer=None,
                gpu=0,
                gamma=0.995,
                lambd=0.95,
                phi=phi,
                value_func_coef=1.0,
                entropy_coef=0.01,
                update_interval=2048,
                minibatch_size=128,
                epochs=10,
                clip_eps=0.2,
                clip_eps_vf=None,
                standardize_advantages=True,
                recurrent=False,
                max_recurrent_sequence_len=None,
                act_deterministically=False,
                value_stats_window=1000,
                entropy_stats_window=1000,
                value_loss_stats_window=100,
                policy_loss_stats_window=100,
                rnd=rnd)
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        reward = 0
        done = False
        while not done:
            action = agent.act_and_train(state, reward)
            state, reward, done, info = env.step(action)
            #if episode % 1 == 0:
                #env.render()
        agent.stop_episode_and_train(state, reward, done)
        if info["flag_get"]:
            print('\nEpisode {0:4d}', episode)
            agent.save('result/ppo-rnd-with-baseline/' + 'well_agent' + str(episode))
        if episode % 100 == 0:
            print('\nEpisode {0:4d}: statistics: {1}'.format(episode, agent.get_statistics()))
    
if __name__ == "__main__":
    main()



