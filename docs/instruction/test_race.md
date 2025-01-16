# 热身赛开发指南
本文档为热身赛（开心消消乐）的开发指南。本文档基于热身赛sdk进行开发，sdk的使用说明可以参考引导文档sdk部分的对应文档。

## stage1: 实现一个能运行的ai
我们首先实现一个能够在平台上运行的agent。

根据sdk的文档，我们需要实现一个ai函数，该函数接收一个EliminationEnv类型的对象，返回一个`[0, 159999]`之间的整数。我们可以使用随机采样实现该函数：

```python
# ai.py
def ai(env: EliminationEnv):
    """
    Args:
        env (EliminationEnv): 每局游戏维护的唯一局面，请不要直接操作该局面

    Returns:
        int: 操作对应的序号，可以使用env的coord_to_num方法将坐标转换为操作序号
    """
    action = random.randint(0, 159999)
    return action
```

随后，我们在main.py中将ai传入controller：

```python
# main.py
from ai import ai
...

if __name__ == "__main__":
    controller = Controller()
    controller.run(ai)
```

这样我们就完成了一个能够在平台上运行的agent。这个agent的策略是：在任意一个回合，从棋盘上选取两个块随机交换。

## stage2：实现一个有策略的ai
在stage1中我们实现了一个简易的agent。但是，该agent的策略有些过于简易，并且完全没有使用环境信息。在本阶段，我们尝试利用env进行agent的策略编写。

机器学习中最简易的算法之一是贪心算法。对于本游戏，由于动作空间始终不发生改变，我们首先可以尝试使用贪心算法，即：遍历所有动作，从中选择当前局面下收益最大的一个。但是由于动作空间很大，在时间限制内几乎不可能计算出所有动作的收益。因此，我们可以通过随机采样来选择一个收益相对较大的动作：

```python
def ai(env: EliminationEnv):
    """
    Args:
        env (EliminationEnv): 每局游戏维护的唯一局面，请不要直接操作该局面

    Returns:
        int: 操作对应的序号，可以使用env的coord_to_num方法将坐标转换为操作序号
    """
    max_reward = 0
    max_action = 0
    
    actions = list(range(160000))
    random.shuffle(actions)

    start = time.time()
    for i in actions:
        env_copy = deepcopy(env)
        board, reward, end = env_copy.step(i)
        if reward > max_reward:
            max_reward = reward
            max_action = i

        if time.time() - start > 0.9:
            break
    return max_action
```
该ai在评测机上，每秒大约能搜索1000条指令，选取到贪心意义下最优动作的概率也提升了1000倍，并且在大部分情况下该解已经是可接受的解。

## stage3：炼丹
stage2已经是一个较优解，但距离最优解仍有距离。首先，对于状态空间的遍历不够彻底。其次，贪心算法在博弈中也很难找到最优解。我们希望使用机器学习方法对最优策略进行求解。

我们以DQN为例讲解使用sdk进行模型训练的过程。我们先使用pytorch实现状动作价值网络：
```python
# dqn.py
class DQN(nn.Module):

    def __init__(self, output_dim):
        super(DQN, self).__init__()
        self.emb = nn.Embedding(5, 32)
        self.conv = nn.Conv2d(32, 64, 11, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 7, padding='same')
        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        self.bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 4, 3, padding='same')
        self.fc = nn.Linear(20*20*4, output_dim)

    def forward(self, x):
        B, W, H = x.shape
        x = self.emb(x.view(B, -1)).view(B, W, H, 32).permute(0, 3, 1, 2)
        x = self.bn(self.conv(x)+self.conv3(x)+self.conv3(x))
        x = nn.functional.leaky_relu(x)
        x = self.conv4(x)
        return self.fc(x.contiguous().view(B, -1))
```
该模型输入为`Batchsize * size * size`的局面张量，输出为`160000`维的Q值向量。

接着我们实现经验回放和epsilon-greedy采样：
```python
# dqn.py
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size))
        return np.stack(state), action, reward, np.stack(next_state), done

    def size(self):
        return len(self.buffer)


def epsilon_greedy_action(q_values, epsilon):
    if random.random() < epsilon:
        return random.randint(0, q_values.size(-1) - 1)
    else:
        return q_values.argmax().item()


def get_action(state_tensor, model, epsilon):
    q_values = model(state_tensor)
    return epsilon_greedy_action(q_values, epsilon)
```
于是我们完成了一个输入局面，输出操作序号的AI。接下来我们需要训练我们的模型：

```python
# train.py
def train_model(
    env,
    self_model,
    enemy_model,
    num_episodes=1000,
    buffer_len=5000,
    batch_size=4,
    gamma=0.99,
    lr=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.2,
    epsilon_decay=10000,
    save_model=False
):
    action_dim = env.action_space.n
    target_q_net = DQN(action_dim).to(device)
    target_q_net.load_state_dict(self_model.state_dict())

    optimizer = optim.Adam(self_model.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=5000)

    epsilon = epsilon_start
    epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        seat = random.randint(0, 1)

        while not done:
            if seat == 0:
                action = get_action(torch.LongTensor(state).unsqueeze(
                    0).to(device), self_model, epsilon)
                next_state, reward, done = env.step(action)

                if not done:
                    enemy_action = get_action(
                        torch.LongTensor(next_state).unsqueeze(0).to(device), enemy_model, epsilon)
                    next_state, enemy_reward, done = env.step(enemy_action)
                    reward -= enemy_reward

            else:
                enemy_action = get_action(
                    torch.LongTensor(state).unsqueeze(0).to(device), enemy_model, epsilon)
                next_state, enemy_reward, done = env.step(enemy_action)
                reward = -enemy_reward

                if not done:
                    action = get_action(torch.LongTensor(next_state).unsqueeze(
                        0).to(device), self_model, epsilon)
                    next_state, reward, done = env.step(action)
                    reward = reward - enemy_reward

            # 存储经验到回放缓冲区
            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done,
            )

            state = next_state
            episode_reward += reward

            # 经验回放训练
            if replay_buffer.size() >= batch_size:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay_buffer.sample(
                    batch_size)

                batch_state_tensor = torch.LongTensor(batch_state).to(device)
                batch_action_tensor = torch.LongTensor(batch_action).to(device)
                batch_reward_tensor = torch.LongTensor(
                    batch_reward).to(device)
                batch_next_state_tensor = torch.LongTensor(
                    batch_next_state).to(device)
                batch_done_tensor = torch.LongTensor(batch_done).to(device)

                q_values = self_model(batch_state_tensor)
                next_q_values = target_q_net(batch_next_state_tensor)

                # 计算Q目标值
                target_q_values = batch_reward_tensor + gamma * \
                    (1 - batch_done_tensor) * next_q_values.max(1)[0]

                # 获取选择的动作的Q值
                q_value = q_values.gather(
                    1, batch_action_tensor.unsqueeze(1)).squeeze(1)

                # 计算损失
                loss = nn.functional.mse_loss(
                    q_value, target_q_values.detach())

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 更新epsilon
            epsilon = max(epsilon_end, epsilon - epsilon_decay_rate)

        # 更新目标网络参数
        if episode % 10 == 0:
            target_q_net.load_state_dict(self_model.state_dict())

        # 如果需要，更新对方网络参数
        if episode % 10 == 0:
            enemy_model.load_state_dict(self_model.state_dict())

        if save_model:
            if episode % 50 == 0:
                torch.save(target_q_net.state_dict(), "model.pt")

        print(
            f"Episode {episode}, Total Reward: {episode_reward}, Epsilon: {epsilon}, loss: {loss.item()}"
        )

    return self_model
```
最后，我们训练模型并保存参数

```python
# train.py
env = EliminationEnv()
action_dim = env.action_space.n
# 初始化Q网络
q_net = DQN(action_dim).to(device)
```

之后，我们修改ai函数并传入controller
```python
# ai.py
model = DQN(160000)
model.load_state_dict("path/to/save/model.pt")
def ai(env: EliminationEnv):
    return get_action(torch.LongTensor(env.observe), model, 0)
```

## stage4：可能的改进
前述三个阶段提供了几个非常简单的设计agent的方法，仍有很多改进空间。例如，即使不采用深度学习方法，也可以使用对抗搜索、蒙特卡洛等方法搜索最优解。又例如，dqn的采样过程可以结合搜索等方法进行长期最优解的搜索。或者，可以采用PPO等方式进行强化学习也可以达到更优的效果。本文仅供大家上手这个比赛，期望大家能够通过改进前述方法或使用更优的方法取得更好的效果。