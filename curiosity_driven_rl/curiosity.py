import gym
from network import *
from tqdm import tqdm
import argparse

def main(test_mode,load):
    env = gym.make('SpaceInvaders-v0')
    env.reset()
    NUM_EPOCHS = 200
    BATCH_SIZE = 128
    DISCOUNT = 0.50
    UPDATE_FREQ = 2000
    EPS_END = 0.05
    EPS_START = 1.4
    EPS_DECAY = 200
    use_greedy = True
    LOAD_PATH = "saves/checkpoint"+str(load)+".pth"
    memory = ReplayMemory(10000)
    n_actions = 6
    screen_height = 84
    screen_width = 84
    test_runs = 10
    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    optimizer = optim.RMSprop(policy_net.parameters())
    cum_rewards = []
    if(test_mode==True):
        if(load ==420):
            target_net.load_state_dict(torch.load('net_weights.pth'))
            target_net.eval()
        else:
            target_net.load_state_dict(torch.load(LOAD_PATH)['weights'])
            target_net.eval()
        for i in range(test_runs):
            action = 0
            cumulated_reward = 0
            current_state = np.zeros((1,4,84,84))
            env.reset()
            count = 0 
            while(True):
                env.render()
                obs,reward,done,_ = env.step(action)
                reward = max(-0.99,min(reward,0.99))
                cumulated_reward +=reward
                obs = preprocess(obs)
                if(count <1):
                    #if frames have not yet been stacked up, use the first observation as the whole state
                    for i in range(4):
                        current_state[0][i] = obs
                else:
                    #replacing the state one by one
                    for i in reversed(range(3)):
                        current_state[0][i+1] = current_state[0][i]
                    current_state[0][0] = obs

                action = torch.argmax(target_net(to_tensor(current_state))).item()
                if(done):
                    print(cumulated_reward)
                    env.close()
                    break
                count +=1
    else:
        if(load!=0 and load != 420):
            checkpoint = torch.load(LOAD_PATH)
            policy_net.load_state_dict(checkpoint['weights'])
            policy_net.eval()
            target_net.load_state_dict(policy_net.state_dict())
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch_start = checkpoint['epoch']
            cum_rewards = checkpoint['rewards']
        else:
            target_net.load_state_dict(policy_net.state_dict())
            epoch_start = 0
            steps_done = 0
        for epoch in range(epoch_start,epoch_start + NUM_EPOCHS):
            action = 0
            current_state = np.zeros((1,4,84,84))
            previous_state = current_state.copy()
            last_previous_state = previous_state.copy()
            last_reward = np.zeros(6)
            current_reward = np.zeros(6)
            last_action = 0
            count = 0
            previous_action = 0
            previous_previous_action = 0
            previous_reward = 0
            previous_previous_reward = 0
            cumulated_reward = 0
            done = False
            while not done:
                #wait for 4 frames to stack up
                obs,reward,done,_ = env.step(action)
                reward = max(-0.99, min(reward, 0.99))
                cumulated_reward += reward
                current_reward[action] = reward
                obs = preprocess(obs)
                cv2.imshow('obs',obs)
                k = cv2.waitKey(1)
                if(k==ord('k')):
                    quit()
                if(count <1):
                    #if frames have not yet been stacked up, use the first observation as the whole state
                    for i in range(4):
                        current_state[0][i] = obs
                    previous_state = current_state.copy()
                    previous_previous_state = previous_state.copy()
                    previous_previous_previous_state = previous_state.copy()
                else:
                    #replacing the state one by one
                    for i in reversed(range(3)):
                        current_state[0][i+1] = current_state[0][i]
                    current_state[0][0] = obs
                    memory.push(previous_previous_previous_state[0],previous_previous_action,0,previous_previous_reward + 0.5*previous_reward + 0.25*current_reward)
                    previous_previous_action = previous_action
                    previous_action = action
                    if(use_greedy):
                        sample = random.random()
                        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                                    math.exp(-1. * steps_done/ EPS_DECAY)
                        if(sample>eps_threshold):
                            with torch.no_grad():
                                action = torch.argmax(policy_net(to_tensor(current_state))).item()
                        else:
                            action = random.randrange(n_actions)
                    action = torch.argmax(policy_net(to_tensor(current_state))).item()
                    previous_previous_previous_state = previous_previous_state
                    previous_previous_state = previous_state
                    previous_state = current_state
                    previous_previous_reward = previous_reward
                    previous_reward = current_reward
                    #=======network optimization time=======#
                    if(count>BATCH_SIZE):
                        sample = memory.sample(BATCH_SIZE)
                        sample = Transition(*zip(*sample))
                        s =  to_tensor(np.array(list(sample.state)))
                        a =  to_tensor(np.array(list(sample.action)))
                        r =  to_tensor(np.array(list(sample.reward)))
                        pred = policy_net(s)
                        target = target_net(s).detach() + 0.9 *r
                        #compute loss
                        optimizer.zero_grad()
                        loss = F.mse_loss(pred, target)
                        loss.backward()
                        for param in policy_net.parameters():
                            param.grad.data.clamp_(-1, 1)
                        optimizer.step()
                            #update our policy network and keep our target network fixed
                count +=1
                steps_done +=1
            #done, so do some update things

            env.reset()
            policy_net.load_state_dict(target_net.state_dict())
            cum_rewards.append(cumulated_reward)
            print(cumulated_reward)
            print("epoch{}".format(epoch))
            torch.save({"weights":target_net.state_dict(),"rewards":cum_rewards,"optimizer":optimizer.state_dict(),"epoch":epoch},"saves/checkpoint"+str(epoch)+".pth")
    return cum_rewards
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="input arguments")
    parser.add_argument('-test_mode',action='store_true',default=False,help="test or not")
    parser.add_argument('--load',type=int,default=0,help='load or not')
    args = parser.parse_args()
    rewards = main(test_mode=args.test_mode,load=args.load)
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.show()


