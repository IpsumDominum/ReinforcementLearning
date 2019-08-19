import gym
from network import *
from tqdm import tqdm
from torchsummary import summary
import argparse

def normalize(tensor):
    return (tensor - tensor.mean())/tensor.std()
def weights_init(self, m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform(m.weight)
def main(test_mode,load):
    env = gym.make('Breakout-v0')
    env.reset()
    NUM_EPOCHS = 1000
    BATCH_SIZE = 512
    DISCOUNT = 0.50
    UPDATE_FREQ = 100
    GAMMA = 0.9
    EPS_END = 0.05
    EPS_START = 0.95
    EPS_DECAY = 10000
    use_greedy = True
    render = True
    LOAD_PATH = "saves/checkpoint"+str(load)+".pth"
    memory = ReplayMemory(10000)
    n_actions = env.action_space.n
    frame_size = 4
    screen_height = 84
    screen_width = 84
    test_runs = 10
    policy_net = DQN(screen_height, screen_width, n_actions,frame_size).to(device)
    target_net = DQN(screen_height, screen_width, n_actions,frame_size).to(device)
    optimizer = optim.RMSprop(policy_net.parameters())
    cum_rewards = []
    random_agent = False
    if(test_mode==True):
        if(load ==420):
            target_net.load_state_dict(torch.load('net_weights.pth'))
            target_net.eval()
        elif(load == 42069):
            target_net.load_state_dict(policy_net.state_dict())
        elif(load == 11111):
            random_agent = True
        else:
            target_net.load_state_dict(torch.load(LOAD_PATH)['target_weights'])
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
                    for i in range(frame_size):
                        current_state[0][i] = obs
                else:
                    #replacing the state one by one
                    for i in reversed(range(frame_size-1)):
                        current_state[0][i+1] = current_state[0][i]
                    current_state[0][0] = obs
                if(random_agent):
                    action = random.randrange(n_actions)
                else:
                    action = torch.argmax(target_net(to_tensor(current_state))).item()
                if(done):
                    print(cumulated_reward)
                    env.close()
                    break
                count +=1
    else:
        if(load!=0 and load != 420):
            checkpoint = torch.load(LOAD_PATH)
            policy_net.load_state_dict(checkpoint['policy_weights'])
            policy_net.eval()
            target_net.load_state_dict(checkpoint['target_weights'])
            target_net.eval()
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch_start = checkpoint['epoch']
            cum_rewards = checkpoint['rewards']
            steps_done= checkpoint['steps_done']
        else:
            target_net.load_state_dict(policy_net.state_dict())
            epoch_start = 0
            steps_done = 0
        for epoch in range(epoch_start,epoch_start + NUM_EPOCHS):
            action = 0
            current_state = np.zeros((1,frame_size,84,84))
            previous_state = current_state.copy()
            last_action = 0
            count = 0
            cumulated_reward = 0
            last_info = 3
            done = False
            while(True):
                #wait for 4 frames to stack up
                obs,reward,done,info = env.step(action)
                reward = max(-0.99, min(reward, 0.99))
                cumulated_reward += reward
                obs = preprocess(obs)
                if(render):
                    cv2.imshow('obs',obs)
                    k = cv2.waitKey(1)
                    if(k==ord('k')):
                        quit()
                    else:
                        pass
                if(count <1):
                    #if frames have not yet been stacked up, use the first observation as the whole state
                    for i in range(frame_size):
                        current_state[0][i] = obs
                    previous_state = current_state.copy()
                else:
                    #replacing the state one by one
                    for i in reversed(range(frame_size-1)):
                        current_state[0][i+1] = current_state[0][i]
                    current_state[0][0] = obs
                    if(not done):
                        memory.push(previous_state[0],np.array([action]),current_state[0],reward)
                    else:
                        memory.push(previous_state[0],np.array([action]),None,0)
                    if(use_greedy):
                        sample = random.random()
                        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                                    math.exp(-1. * count/ EPS_DECAY)
                        if(sample>eps_threshold):
                            with torch.no_grad():
                                action_pred = policy_net(to_tensor(current_state))
                                action = torch.argmax(action_pred[0]).item()
                        else:
                            action = random.randrange(n_actions)
                    else:
                        action = torch.argmax(policy_net(to_tensor(current_state))).item()
                    previous_state = current_state
                    #=======network optimization time=======#
                    if(len(memory.memory)>BATCH_SIZE):
                        sample = memory.sample(BATCH_SIZE)
                        sample = Transition(*zip(*sample))
                        s =  to_tensor(np.array(list(sample.state)))
                        a =  to_tensor(np.array(list(sample.action))).type(torch.cuda.LongTensor)
                        r =  to_tensor(np.array(list(sample.reward)))
                        
                        n_s = to_tensor(np.array([s for s in sample.next_state
                                                  if s is not None]))
                        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                                sample.next_state)), device=device, dtype=torch.uint8)
                        pred = policy_net(s).gather(1,a).squeeze(1)
                        next_state_values = torch.zeros(BATCH_SIZE, device=device)
                        target = target_net(n_s).detach().max(1)[0]
                        next_state_values[non_final_mask] = GAMMA * target
                        next_state_values += r
                        #compute loss
                        loss = F.smooth_l1_loss(pred, next_state_values)
                        optimizer.zero_grad()
                        loss.backward()
                        for param in policy_net.parameters():
                            param.grad.data.clamp_(-1, 1)
                            optimizer.step()
                            #update our policy network and keep our target network fixed
                count +=1
                steps_done +=1
                if(done):
                    
                    target_net.load_state_dict(policy_net.state_dict())
                    #compute discounted rewards overtime
                    env.reset()
                    cum_rewards.append(cumulated_reward)
                    print(cumulated_reward)
                    print(count)
                    print("epoch{}".format(epoch))
                    break
        torch.save({"target_weights":target_net.state_dict(),"policy_weights":policy_net.state_dict(),"rewards":cum_rewards,"optimizer":optimizer.state_dict(),"epoch":epoch,"steps_done":steps_done},"checkpoint.pth")
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


