# 吃豆人开发指南
本文档为吃豆人的开发指南。本文档基于热身赛sdk进行开发，sdk的使用说明可以参考引导文档sdk部分的对应文档。

## stage1: 实现一个能运行的ai
### 吃豆人
我们首先实现一个能够在平台上运行的agent。

根据sdk的文档，我们需要实现一个ai函数，该函数接收一个GameState类型的对象，返回一个<b>长度为1</b>的数组，数组元素为`[0, 4]`之间的整数。我们可以使用随机采样实现该函数：
```python
# ai.py
def ai_func(game_state: GameState):
    """
    Args:
        game_state (GameState): 当前的游戏局面对象

    Returns:
        list: 包含一个操作序号的数组，范围在[0, 4]之间
    """
    action = random.randint(0, 4)
    return [action]
```

### 幽灵
我们首先实现一个能够在平台上运行的agent。

根据sdk的文档，我们需要实现一个ai函数，该函数接收一个GameState类型的对象，返回一个<b>长度为3</b>的数组，数组元素为`[0, 4]`之间的整数。我们可以使用随机采样实现该函数：
```python
# ai.py
def ai(game_state: GameState):
    """
    Args:
        game_state (GameState): 当前的游戏局面对象

    Returns:
        list: 包含一个操作序号的数组，范围在[0, 4]之间
    """
    action = [random.randint(0, 4) for _ in range(3)]
    return action
```
此时我们已经实现了一个能运行的ai，但是很明显随机移动的ai不足以取得好的效果，下面我们实现一个有策略的ai。
## stage2：实现一个有策略的ai
我们已经在gamedata.py文件中定义了很多封装与抽象，你可以直接使用来简化你的代码逻辑。
### 吃豆人
我们可以实现一个基于状态机的AI来控制吃豆人。这个AI会根据当前游戏局面，动态切换不同的行为状态，并使用评分函数来选择最优的移动方向。

主要的状态包括:

- COLLECT: 收集普通豆子模式，在安全时主动寻找并吃掉普通豆子
- ESCAPE: 逃离幽灵模式，当检测到附近有幽灵威胁时，优先考虑逃离 
- BONUS: 收集特殊豆子模式，发现高价值的特殊道具时会优先获取
- GETOUT: 逃离传送门模式，在合适的时机通过传送门进入下一关

AI的主要功能包括:

1. 状态更新
- 根据幽灵距离、特殊豆子位置、护盾状态等因素动态切换状态
- 在不同状态下使用不同的评分权重

2. 路径规划
- 使用A*算法进行路径搜索
- 考虑墙壁、传送门等障碍物

3. 位置评估
- 计算幽灵威胁程度
- 评估周围豆子的价值
- 考虑历史访问记录避免重复移动
- 根据当前状态调整评分权重

4. 移动决策
- 评估所有可能的移动方向
- 选择评分最高的移动方向
- 更新历史记录

完整的代码实现如下:
```python
from core.gamedata import *
import numpy as np
from enum import Enum
from collections import deque

class AIState(Enum):
    COLLECT = "COLLECT"  # 收集普通豆子模式
    ESCAPE = "ESCAPE"  # 逃离幽灵模式
    BONUS = "BONUS"  # 收集特殊豆子模式
    GETOUT = "GETOUT"  # 逃离传送门模式

class PacmanAI:
    def __init__(self):
        self.board_size = None
        self.current_state = AIState.COLLECT
        self.path = []
        # 历史记录
        self.history = deque(maxlen=20)
        self.init_bean_num = 0

        # 动态参数
        self.GHOST_DANGER_DISTANCE = 5
        self.BONUS_BEAN_PRIORITY = 2.0

        # 状态权重
        self.weights = {
            AIState.COLLECT: {"ghost": 1.0, "bean": 1.5, "bonus": 1.5},
            AIState.ESCAPE: {"ghost": 3.0, "bean": 0.8, "bonus": 1.0},
            AIState.BONUS: {"ghost": 0.8, "bean": 1.5, "bonus": 2.0},
            AIState.GETOUT: {"ghost": 1, "bean": 1, "bonus": 1},
        }

        # 特殊豆子价值
        self.bean_values = {
            3: 4.0,  # BONUS_BEAN
            4: 3.0,  # SPEED_BEAN
            5: 2.5,  # MAGNET_BEAN
            6: 3.0,  # SHIELD_BEAN
            7: 2.5,  # DOUBLE_BEAN
        }

    def count_remaining_bean(self, game_state: GameState):
        """计算剩余豆子数量"""
        cnt = 0
        for i in range(game_state.board_size):
            for j in range(game_state.board_size):
                if game_state.board[i][j] in range(2, 8):
                    cnt += 1

        return cnt

    def point_to_vector_projection_distance(self, point, vector_start, vector_end):
        """计算点到向量的投影距离"""
        vector = vector_end - vector_start
        point_vector = point - vector_start
        vector_length = np.linalg.norm(vector)

        if vector_length == 0:
            return np.linalg.norm(point_vector)

        vector_unit = vector / vector_length
        projection_length = np.dot(point_vector, vector_unit)
        projection_vector = vector_unit * projection_length
        projection_point = vector_start + projection_vector
        return np.linalg.norm(point - projection_point)

    def can_getout_before_ghosts(self, game_state: GameState):
        """判断是否能在幽灵到达前到达传送门"""
        pacman_pos = np.array(game_state.pacman_pos)
        portal_pos = np.array(game_state.portal_coord)
        ghosts_pos = np.array(game_state.ghosts_pos)

        dist_to_portal = np.linalg.norm(pacman_pos - portal_pos)
        ghosts_projection_dist_to_catch = [
            self.point_to_vector_projection_distance(ghost_pos, pacman_pos, portal_pos)
            for ghost_pos in ghosts_pos
        ]

        return dist_to_portal < min(ghosts_projection_dist_to_catch) - 1

    def manhattan_distance(self, pos1, pos2):
        """计算曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def update_state(self, game_state: GameState):
        """更新游戏状态"""
        pacman_pos = np.array(game_state.pacman_pos)
        ghosts_pos = np.array(game_state.ghosts_pos)
        # 计算威胁程度
        ghost_distances = [
            len(
                self.a_star_search(
                    np.array(pacman_pos), np.array(ghost_pos), game_state
                )
            )
            for ghost_pos in game_state.ghosts_pos
        ]
        min_ghost_distance = min(ghost_distances)
        # 寻找特殊豆子
        special_bean = self.find_nearest_special_bean(game_state)
        # 检查是否有护盾状态
        has_shield = game_state.pacman_skill_status[Skill.SHIELD.value] > 0
        # 状态机转换逻辑
        if min_ghost_distance < self.GHOST_DANGER_DISTANCE and not has_shield:
            # 如果可以在幽灵到达前到达传送门，优先选择GETOUT
            if (
                game_state.level < 3
                and self.can_getout_before_ghosts(game_state)
                and game_state.portal_available
                and self.count_remaining_bean(game_state) < self.init_bean_num * 0.5
            ):
                self.current_state = AIState.GETOUT
            else:
                self.current_state = AIState.ESCAPE
        elif (
            game_state.level < 3
            and game_state.portal_available
            and self.count_remaining_bean(game_state) < self.init_bean_num * 0.5
        ):
            self.current_state = AIState.GETOUT
        elif special_bean and special_bean[1] < 8:
            self.current_state = AIState.BONUS
        else:
            self.current_state = AIState.COLLECT

    def find_nearest_special_bean(self, game_state):
        """寻找最近的特殊豆子"""
        pacman_pos = np.array(game_state.pacman_pos)
        special_beans = []

        for i in range(game_state.board_size):
            for j in range(game_state.board_size):
                bean_type = game_state.board[i][j]
                if bean_type >= 3 and bean_type <= 7:  # 特殊豆子
                    pos = np.array([i, j])
                    dist = np.linalg.norm(pacman_pos - pos)
                    value = self.bean_values[bean_type]
                    score = value / (dist + 1)  # 考虑距离和价值的综合评分
                    special_beans.append((pos, dist, score))

        if special_beans:
            # 按综合评分排序
            best_bean = max(special_beans, key=lambda x: x[2])
            return (best_bean[0], best_bean[1])
        return None

    def a_star_search(self, start: np.ndarray, goal: np.ndarray, game_state: GameState):
        """A*搜索路径"""
        open_set = set()
        open_set.add(tuple(start))
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): self.manhattan_distance(start, goal)}

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float("inf")))
            if current == tuple(goal):
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            open_set.remove(current)
            for direction, _ in self.get_valid_moves(list(current), game_state):
                neighbor = tuple(direction)
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.manhattan_distance(
                        neighbor, goal
                    )
                    if neighbor not in open_set:
                        open_set.add(neighbor)

        return []

    def evaluate_position(self, pos, game_state):
        """评估位置的价值"""
        pacman_pos = np.array(game_state.pacman_pos)
        weights = self.weights[self.current_state]
        ghost_distances = [
            len(self.a_star_search(pos, np.array(ghost_pos), game_state))
            for ghost_pos in game_state.ghosts_pos
        ]
        min_ghost_distance = min(ghost_distances)
        ghost_score = (-4) * weights["ghost"] / (min_ghost_distance + 1)

        # 计算周围豆子的价值
        bean_value = 0
        scan_range = 5
        for dx in range(-scan_range, scan_range + 1):
            for dy in range(-scan_range, scan_range + 1):
                new_x, new_y = int(pos[0] + dx), int(pos[1] + dy)
                if (
                    0 <= new_x < game_state.board_size
                    and 0 <= new_y < game_state.board_size
                ):
                    bean_type = game_state.board[new_x][new_y]
                    if bean_type in BEANS_ITERATOR:  # 有豆子
                        distance = abs(dx) + abs(dy)
                        if bean_type in SPECIAL_BEANS_ITERATOR:  # 特殊豆子
                            bean_value += self.bean_values[bean_type] / (distance + 1)
                        elif bean_type == Space.REGULAR_BEAN.value:  # 普通豆子
                            bean_value += 1 / (distance + 1)

        # 避免重复访问
        pos_tuple = tuple(pos)
        repeat_penalty = 0
        visit_count = self.history.count(pos_tuple)
        repeat_penalty = -8 * visit_count
        final_score = ghost_score + bean_value * weights["bean"] + repeat_penalty

        # 如果是逃离模式，确保远离幽灵
        if self.current_state == AIState.ESCAPE:
            if min_ghost_distance < self.GHOST_DANGER_DISTANCE:
                final_score -= (self.GHOST_DANGER_DISTANCE - min_ghost_distance) * 10

        # 如果是GETOUT模式，确保尽快到达传送门
        if self.current_state == AIState.GETOUT:
            portal_pos = np.array(game_state.portal_coord)
            dist_to_portal = np.linalg.norm(pos - portal_pos)
            final_score += 30 / (dist_to_portal + 1)
        return final_score

    def get_valid_moves(self, pos, game_state):
        """获取有效的移动方向"""
        moves = []
        directions = [
            (np.array(list(Update.UP.value)), Direction.UP.value),  # UP
            (np.array(list(Update.LEFT.value)), Direction.LEFT.value),  # LEFT
            (np.array(list(Update.DOWN.value)), Direction.DOWN.value),  # DOWN
            (np.array(list(Update.RIGHT.value)), Direction.RIGHT.value),  # RIGHT
        ]
        for direction, move_num in directions:
            new_pos = pos + direction
            if self.is_valid_position(new_pos, game_state):
                moves.append((new_pos, move_num))
        return moves

    def is_valid_position(self, pos, game_state):
        """检查位置是否有效"""
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < game_state.board_size and 0 <= y < game_state.board_size:
            if self.current_state != AIState.GETOUT:
                if game_state.board[x][y] == Space.PORTAL.value:
                    return False
            if game_state.board[x][y] != Space.WALL.value:
                return True
        return False

    def choose_move(self, game_state: GameState):
        """选择移动方向"""
        # 初始化
        if game_state.round == 1:
            self.init_bean_num = self.count_remaining_bean(game_state)
        self.board_size = game_state.board_size
        self.update_state(game_state)
        pacman_pos = np.array(game_state.pacman_pos)
        valid_moves = self.get_valid_moves(pacman_pos, game_state)
        # 评估每个可能的移动
        move_scores = []
        for new_pos, move_num in valid_moves:
            score = self.evaluate_position(new_pos, game_state)
            move_scores.append((score, move_num))
        # 选择最佳移动
        if move_scores:
            best_score, best_move = max(move_scores, key=lambda x: x[0])
            # 更新历史记录
            self.history.append(tuple(pacman_pos))
            return [best_move]
        return [Direction.STAY.value]  # 默认停留

ai_func = PacmanAI().choose_move
__all__ = ["ai_func"]
```
注意：按照要求，我们的choose_move函数返回一个长度为1的列表，并自定义导出名称为ai_func。
### 幽灵
因为幽灵控制者能够控制多个幽灵，所以可以对不同的幽灵构造不同的策略，达到围堵吃豆人的目的。

幽灵AI的主要策略包括:

1. 基本功能
- 使用曼哈顿距离计算位置间距离
- 获取有效移动方向
- A*寻路算法实现路径规划
- 记录历史位置避免重复移动

2. 三个幽灵的不同策略

幽灵1(Ghost 0):
- 直接追击模式
- 使用A*算法寻找最短路径追击吃豆人
- 作为主要追击者

幽灵2(Ghost 1): 
- 预测截击模式
- 预测吃豆人移动方向
- 提前移动到预测位置拦截
- 与幽灵1重合时随机移动

幽灵3(Ghost 2):
- 协同包围模式 
- 参考幽灵1位置进行包围
- 尝试切断吃豆人逃跑路线
- 与其他幽灵重合时随机移动

3. 移动决策
- 距离吃豆人很近(<=3)时直接追击
- 距离较远时根据不同幽灵策略行动
- 使用历史记录避免重复移动
- 在无法到达目标时选择最近路径

4. 关键实现
- 使用A*算法进行路径规划
- 计算曼哈顿距离评估位置
- 记录历史位置防止重复
- 不同幽灵协同配合

通过以上策略,三个幽灵可以有效配合围堵吃豆人。

```python
from core.gamedata import *
import random
import numpy as np

def parse(x: tuple):
    if x == (0, 0):
        return Direction.STAY.value
    if x == (1, 0):
        return Direction.UP.value
    if x == (-1, 0):
        return Direction.DOWN.value
    if x == (0, 1):
        return Direction.RIGHT.value
    if x == (0, -1):
        return Direction.LEFT.value


class GhostAI:
    def __init__(self):
        self.position_history = {0: [], 1: [], 2: []}
        self.history_length = 5

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_valid_moves(self, pos, game_state):
        valid_moves = []
        directions = [
            ([1, 0], 1),  # UP
            ([-1, 0], 3),  # DOWN
            ([0, -1], 2),  # LEFT
            ([0, 1], 4),  # RIGHT
        ]

        for direction, move_value in directions:
            new_pos = [pos[0] + direction[0], pos[1] + direction[1]]
            if (
                0 <= new_pos[0] < game_state.board_size
                and 0 <= new_pos[1] < game_state.board_size
                and game_state.board[new_pos[0]][new_pos[1]] != 0
            ):
                valid_moves.append((new_pos, move_value))
        return valid_moves

    def a_star_search(self, start: np.ndarray, goal: np.ndarray, game_state: GameState):
        open_set = set()
        open_set.add(tuple(start))
        came_from = {}

        g_score = {tuple(start): 0}
        f_score = {tuple(start): self.manhattan_distance(start, goal)}

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float("inf")))
            if current == tuple(goal):
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            open_set.remove(current)
            for direction, _ in self.get_valid_moves(list(current), game_state):
                neighbor = tuple(direction)
                tentative_g_score = g_score[current] + 1

                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.manhattan_distance(
                        neighbor, goal
                    )
                    if neighbor not in open_set:
                        open_set.add(neighbor)

        return []

    def calculate_stagnation_penalty(self, new_pos, ghost_id):
        if not self.position_history[ghost_id]:
            return 0
        repeat_count = sum(
            1
            for pos in self.position_history[ghost_id]
            if pos[0] == new_pos[0] and pos[1] == new_pos[1]
        )
        return repeat_count * 2

    def update_history(self, ghost_id, new_pos):
        self.position_history[ghost_id].append(new_pos)
        if len(self.position_history[ghost_id]) > self.history_length:
            self.position_history[ghost_id].pop(0)

    def choose_moves(self, game_state: GameState):
        moves = []
        pacman_pos = game_state.pacman_pos

        for ghost_id in range(3):
            current_pos = game_state.ghosts_pos[ghost_id]
            valid_moves = self.get_valid_moves(current_pos, game_state)

            if not valid_moves:
                moves.append(Direction.STAY.value)
                continue

            # 计算到吃豆人的距离
            a_star_path = self.a_star_search(current_pos, pacman_pos, game_state)
            distance_to_pacman = len(a_star_path) if a_star_path else float("inf")

            # 如果距离很近（比如小于3），直接追击
            if distance_to_pacman <= 3:
                best_move = (
                    a_star_path[0],
                    parse(
                        (
                            a_star_path[0][0] - current_pos[0],
                            a_star_path[0][1] - current_pos[1],
                        )
                    ),
                )
            else:
                # 距离较远时使用更复杂的策略
                target_pos = pacman_pos
                # 不同幽灵的策略
                if ghost_id == 0:
                    # 通过A*寻路算法直接追击
                    pass
                elif ghost_id == 1:
                    if (
                        current_pos[0] == game_state.ghosts_pos[0][0]
                        and current_pos[1] == game_state.ghosts_pos[0][1]
                    ):
                        # 第二个幽灵与第一个幽灵重合时，随机移动
                        best_move = random.choice(valid_moves)
                        self.update_history(ghost_id, best_move[0])
                        moves.append(best_move[1])
                        continue
                    # 预测吃豆人移动方向
                    dx = pacman_pos[0] - current_pos[0]
                    dy = pacman_pos[1] - current_pos[1]
                    predicted_x = pacman_pos[0] + (1 if dx > 0 else -1 if dx < 0 else 0)
                    predicted_y = pacman_pos[1] + (1 if dy > 0 else -1 if dy < 0 else 0)
                    if (
                        0 <= predicted_x < game_state.board_size
                        and 0 <= predicted_y < game_state.board_size
                        and game_state.board[predicted_x][predicted_y] != 0
                    ):
                        target_pos = np.array([predicted_x, predicted_y])
                else:
                    if (
                        current_pos[0] == game_state.ghosts_pos[0][0]
                        and current_pos[1] == game_state.ghosts_pos[0][1]
                    ) or (
                        current_pos[0] == game_state.ghosts_pos[1][0]
                        and current_pos[1] == game_state.ghosts_pos[1][1]
                    ):

                        # 第三个幽灵与第一个或第二个幽灵重合时，随机移动
                        best_move = random.choice(valid_moves)
                        self.update_history(ghost_id, best_move[0])
                        moves.append(best_move[1])
                        continue
                    # 尝试切断路线
                    other_ghost = game_state.ghosts_pos[0]  # 使用第一个幽灵作为参考
                    dx = pacman_pos[0] - other_ghost[0]
                    dy = pacman_pos[1] - other_ghost[1]
                    intercept_x = pacman_pos[0] + (1 if dx > 0 else -1 if dx < 0 else 0)
                    intercept_y = pacman_pos[1] + (1 if dy > 0 else -1 if dy < 0 else 0)
                    intercept_x = max(0, min(intercept_x, game_state.board_size - 1))
                    intercept_y = max(0, min(intercept_y, game_state.board_size - 1))
                    if game_state.board[intercept_x][intercept_y] != 0:
                        target_pos = np.array([intercept_x, intercept_y])

                path = self.a_star_search(current_pos, target_pos, game_state)

                if path:
                    best_move = (
                        path[0],
                        parse(
                            (path[0][0] - current_pos[0], path[0][1] - current_pos[1])
                        ),
                    )
                else:
                    best_move = min(
                        valid_moves,
                        key=lambda x: self.manhattan_distance(x[0], pacman_pos),
                    )

            self.update_history(ghost_id, best_move[0])
            moves.append(best_move[1])

        return moves


ai_func = GhostAI().choose_moves
__all__ = ["ai_func"]
```
## stage3：炼丹
stage2已经是一个较优解，但距离最优解仍有距离。首先，对于状态空间的遍历不够彻底。其次，固定在博弈中也很难事最优解。我们希望使用机器学习方法对最优策略进行求解。

我们以DQN为例介绍使用Pacman SDK中的Gym Environment进行深度强化学习训练的流程。

首先我们设计对局面的表示。此处的实现为将棋盘、幽灵位置、吃豆人位置和出口位置构成的四通道二位张量和将游戏阶段、回合数、棋盘大小和出口是否激活重复十次构成的一维向量作为一个局面的表示。该表示可以通过`state_dict_to_tensor`方法由step函数返回的state获得。

接着使用pytorch实现动作价值网络（此处仅展示Pacman动作价值网络的实现）：
```python
# model.py
class PacmanNet(nn.Module):
    def __init__(self, input_channel_num, num_actions, extra_size):
        super().__init__()
        self.channels = input_channel_num
        self.embeddings = nn.ModuleList(
            [nn.Embedding(9, 16) for _ in range(input_channel_num)])
        self.conv1 = nn.Conv2d(64, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)

        self.encoder = nn.Linear(extra_size, 64)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_actions)

    def forward(self, x, y):
        B, C, H, W = x.shape
        embedded_channels = []
        for i in range(self.channels):
            flattened_channel = x[:, i, :, :].view(B, -1).long()
            embedded_channel = self.embeddings[i](flattened_channel)
            embedded_channel = embedded_channel.view(
                B, 16, H, W)
            embedded_channels.append(embedded_channel)
        # Concatenate along the channel dimension
        x = torch.cat(embedded_channels, dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn(x)
        x = F.relu(self.conv3(x))
        y = F.sigmoid(self.encoder(y))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x+y))
        return self.fc2(x)
```

随后实现epsilon-greedy采样：
```python
# train.py

# epsilon-greedy policy for rollout
def select_action_ghost(state, extra, epsilon, policy_net):
    if np.random.rand() < epsilon:
        return np.random.randint(size=3, low=0, high=4)
    else:
        with torch.no_grad():
            values = policy_net(
                state.to(device), extra.to(device)).reshape(-1, 5)
            # print(f"{values.shape=}")
            return torch.argmax(values, dim=1).cpu().numpy()


def select_action_pacman(state, extra, epsilon, policy_net):
    if np.random.rand() < epsilon:
        return np.random.randint(low=0, high=4)
    else:
        with torch.no_grad():
            return torch.argmax(policy_net(state.to(device), extra.to(device))).cpu().item()

```

最后我们实现Q-learning算法：
```python
# train.py

# training iteration
if __name__ == "__main__":
    num_episodes = 1000
    epsilon = EPSILON_START
    for episode in range(num_episodes):
        state = env.reset(mode="local")
        state, extra = state_dict_to_tensor(state)
        # print(state.shape, extra.shape)

        for t in range(1000):
            action1 = select_action_pacman(state, extra, epsilon, policy_net_pacman)
            action2 = select_action_ghost(
                state, extra, epsilon, policy_net_ghost)
            next_state, reward1, reward2, done, _ = env.step(action1, action2)
            env.render('local')
            next_state, next_extra = state_dict_to_tensor(next_state)
            # next_state = torch.tensor(
            # next_state, dtype=torch.float32).unsqueeze(0)
            reward1 = torch.tensor([reward1], dtype=torch.float32)
            reward2 = torch.tensor([reward2], dtype=torch.float32)
            # print(next_state.shape, next_extra.shape)
            print(reward1.item(), reward2.tolist())


            memory.append(
                Transition(
                    state,
                    extra,
                    torch.tensor([[action1]]),
                    torch.tensor([[action2]]),
                    next_state,
                    next_extra,
                    reward1,
                    reward2,
                )
            )
            if len(memory) > MEMORY_SIZE:
                memory.pop(0)

            state = next_state

            optimize_model()

            if done:
                break

        if episode % TARGET_UPDATE == 0:
            target_net_pacman.load_state_dict(policy_net_pacman.state_dict())
            target_net_ghost.load_state_dict(policy_net_ghost.state_dict())
            torch.save(policy_net_pacman.state_dict(), "pacman.pth")
            torch.save(policy_net_ghost.state_dict(), "ghost.pth")

        epsilon = max(EPSILON_END, EPSILON_START - episode / EPSILON_DECAY)

```

## stage4：可能的改进
前述三个阶段提供了几个非常简单的设计 agent 的方法，但仍有很多改进空间。例如，即使不采用深度学习方法，也可以使用对抗搜索（如 Minimax 算法）和蒙特卡洛树搜索（MCTS）等方法来搜索最优解。这些方法在经典的博弈问题中表现优异，可以为游戏提供有效的策略。

另外，除了简单地使用DQN，还可以尝试其他强化学习算法，如 Proximal Policy Optimization (PPO)、Advantage Actor-Critic (A2C) 等。这些算法在处理复杂策略时表现更好，可能会带来更优的效果。

此外，Pacman 和 Ghosts 是对立的多智能体系统。可以研究多智能体强化学习（MARL）方法，使得各个智能体之间能够更好地协作或对抗，从而提升整体表现。

本文仅供大家上手这个比赛，期望大家能够通过改进前述方法或使用更优的方法取得更好的效果。