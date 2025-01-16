# 吃豆人 SDK 引导

本文档是吃豆人和幽灵的 SDK 使用说明

吃豆人 SDK 仓库：[https://github.com/PacMan-Logic/PacmanSDK-python](https://github.com/PacMan-Logic/PacmanSDK-python)

幽灵 SDK 仓库：[https://github.com/PacMan-Logic/GhostsSDK-python](https://github.com/PacMan-Logic/GhostsSDK-python)

## SDK的下载及配置
1. 安装并配置git
2. 克隆远程仓库到本地（在终端输入）: `git clone <repository_url>`
3. 配置sdk中的core（进入SDK所在目录并在终端输入）: 
`git submodule init` `git submodule update`
4. 后续更新core（在终端输入）: `git submodule update --remote core`

## 代码结构

SDK 由 GYM 环境类、游戏控制器和 AI 函数组成。

### GYM 环境类（PacmanEnv）

GYM 环境类（PacmanEnv）维护了游戏局面的全量信息供 AI 调用。该类提供如下接口：

#### reset

* Returns: return_dict(dict)

reset 函数在每关开始时由 judger 调用。玩家将进入一个新的地图，吃豆人和三个幽灵会随机生成在地图的四个角落。该函数会返回一个包含新地图信息的 JSON 字符串。随后，judger 会将这个 JSON 字符串编码并发送给 AI。

#### ai_reset

* Args: dict(dict)

在棋盘改变时解码 judger 发送的初始化信息，保证 ai 的 PacmanEnv 环境类与后端的 PacmanEnv 环境类的一致性。

#### render

* Args: mode(string) 为"logic"或"local"

* Returns: return_dict(dict)

传入 mode="local"时可在本地终端生成地图，供选手调试使用。

#### step

* Args: pacmanAction(int), ghostAction(List[int])

* Returns: (level_change(bool), eat_all_beans(bool))(tuple)  level_change表示是否切换到下一关，eat_all_beans表示本关是否吃完全部的豆子

step 函数是环境更新的主函数，处理游戏逻辑，按照选手的输入更新游戏状态。

> 具体逻辑的实现请参考 sdk 中 GymEnvironment.py 中的注释

### 游戏控制器（Controller）

#### 初始化

初始化时 judger 向 0 号玩家发送字符串"0"，向 1 号玩家发送字符串"1"，0 号玩家先发送操作消息，1 号玩家后发送操作消息。

#### run

run 函数首先判断是否进入新的 level，若进入新的 level 则读取 judger 传来的初始化信息并实例化整局游戏唯一的环境实例。

若当前玩家为<b>0 号玩家</b>，则先调用传入的 ai 函数，并将 ai 函数返回的操作序号传给 judger

```py
pacman_op(self.env,ai) # 当前玩家为吃豆人，则ai应该返回一个含1个元素的数组
```

或

```py
ghosts_op(self.env,ai) # 当前玩家为幽灵，则ai返回一个含3个元素的数组
```

然后读取对方已经操作的消息

```py
get_info = input() # 并不能读取到对方操作的具体内容，只是标记对方已经发送操作
```

最后读取 judger 传来的幽灵和吃豆人的操作信息，更新整局游戏唯一的环境实例。

若当前玩家为<b>1 号玩家</b>，则先读取对方已经操作的消息

```py
get_info = input() # 并不能读取到对方操作的具体内容，只是标记对方已经发送操作
```

然后调用传入的 ai 函数，并将 ai 函数返回的操作序号传给 judger

```py
pacman_op(self.env,ai) # 当前玩家为吃豆人，则ai应该返回一个含1个元素的数组
```

或

```py
ghosts_op(self.env,ai) # 当前玩家为幽灵，则ai返回一个含3个元素的数组
```

最后读取 judger 传来的幽灵和吃豆人的操作信息，更新整局游戏唯一的环境实例。

**注意若 AI 发送了不合理的操作号（比如 5、-1 等）则会被视为不进行移动**

### AI 函数

一个输入为 GameState 类实例，输出为操作数组的函数。**注意，请不要在函数中直接对传入的环境实例作修改。请使用 deepcopy 复制**

## 选手可获取的内容及获取方式

### 获取关卡号

```py
gamestate.level
```

为`int`类型的值

表示当前关卡号

### 获取轮数

```py
gamestate.round
```

为`int`类型的值

表示当前关卡进行到的轮数

### 获取棋盘相关信息

```py
gamestate.board_size
```

为`int`类型的值

表示当前棋盘大小

```py
gamestate.board
```

为`np.ndarray`类型的值

表示棋盘，棋盘中元素

```
0代表墙
1代表空地
2代表知识金币
3代表双倍荣耀币
4代表疾风之翼
5代表智引磁石
6代表护学之盾
7代表智慧圣典
8代表传送门
```

### 获取吃豆人相关信息

```py
gamestate.pacman_skill_status
```

为`list[int]`类型的值

表示吃豆人当前拥有的技能，数组共 4 个元素，分别表示<b>DOUBLE_SCORE 技能、SPEED_UP 技能、MAGNET 技能的剩余轮数和当前拥有的 SHIELD 的数量</b>

```py
gamestate.pacman_pos
```

为`np.ndarray[int]`类型的值

长度为 2， `gamestate.pacman_pos[0]` 和 `gamestate.pacman_pos[1]` 分别表示吃豆人的横纵坐标

```py
gamestate.pacman_score
```

为`int`类型的值

表示当前吃豆人的得分

### 获取幽灵相关信息

```py
gamestate.ghosts_pos
```

为`np.ndarray[np.ndarray[int]]`类型的值

长度为 3，表示三个幽灵的坐标

```py
gamestate.ghosts_score
```

为`int`类型的值

表示当前幽灵的得分

### 获取传送门相关信息

```py
gamestate.portal_available
```

为`bool`类型的值

表示当前传送门是否开启

```py
gamestate.portal_coord
```

为`np.ndarray[int]`类型的值

长度为 2， `gamestate.portal_coord[0]` 和 `gamestate.portal_coord[1]` 分别表示传送门的横纵坐标

### 获取空间信息

```py
gamestate.space_info
```

为`dict`类型的值

包含`observation_space`、`pacman_action_space`、`ghost_action_space`三个键，分别表示空间的观察空间、吃豆人的动作空间和幽灵的动作空间

示例：

```python
{
    "observation_space": spaces.MultiDiscrete(np.ones((20, 20)) * SPACE_CATEGORY),
    "pacman_action_space": spaces.Discrete(5),
    "ghost_action_space": spaces.MultiDiscrete(np.ones(3) * 5)
}
```