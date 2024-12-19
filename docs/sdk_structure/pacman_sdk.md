# 吃豆人SDK引导
本文档是吃豆人和幽灵的SDK使用说明

吃豆人SDK仓库：[https://github.com/PacMan-Logic/PacmanSDK-python](https://github.com/PacMan-Logic/PacmanSDK-python)

幽灵SDK仓库：[https://github.com/PacMan-Logic/GhostsSDK-python](https://github.com/PacMan-Logic/GhostsSDK-python)

## 代码结构

SDK由GYM环境类、游戏控制器和AI函数组成。

### 将sdk下载到本地及配置
1. 安装并配置git
2. 克隆远程仓库到本地（在终端输入）:`git clone <repository_url>`
3. 配置sdk中的core（在终端输入）:`git submodule update --remote core`

### GYM环境类（PacmanEnv）
GYM环境类（PacmanEnv）维护了游戏局面的全量信息供AI调用。该类提供如下接口：
#### reset
reset 函数在每关开始时由 judger 调用。玩家将进入一个新的地图，吃豆人和三个幽灵会随机生成在地图的四个角落。该函数会返回一个包含新地图信息的 JSON 字符串。随后，judger 会将这个 JSON 字符串编码并发送给 AI。

#### ai_reset
此函数由 AI 调用，用于解码 judger 发送的信息并初始化本地地图。

#### render
传入mode="local"时可在本地终端生成地图，供选手调试使用。

#### step
TODO(zyc): step说明

### Controller

#### 初始化
初始化时 judger 向0号玩家发送字符串"0"，向1号玩家发送字符串"1"，0号玩家先发送操作消息，1号玩家后发送操作消息。

#### run
run函数首先判断是否进入新的level，若进入新的level则读取judger传来的初始化信息并实例化整局游戏唯一的环境实例。

若当前玩家为<b>0号玩家</b>，则先调用传入的ai函数，并将ai函数返回的操作序号传给judger
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
最后读取judger传来的幽灵和吃豆人的操作信息，更新整局游戏唯一的环境实例。


若当前玩家为<b>1号玩家</b>，则先读取对方已经操作的消息
```py
get_info = input() # 并不能读取到对方操作的具体内容，只是标记对方已经发送操作
```
然后调用传入的ai函数，并将ai函数返回的操作序号传给judger
```py
pacman_op(self.env,ai) # 当前玩家为吃豆人，则ai应该返回一个含1个元素的数组
```
或
```py
ghosts_op(self.env,ai) # 当前玩家为幽灵，则ai返回一个含3个元素的数组
```

最后读取judger传来的幽灵和吃豆人的操作信息，更新整局游戏唯一的环境实例。

**注意若AI发送了不合理的操作号（比如5、-1等）则会被视为不进行移动**

### AI函数
一个输入为GameState类实例，输出为操作数组的函数。**注意，请不要在函数中直接对传入的环境实例作修改。请使用deepcopy复制**

## 选手可获取的内容及获取方式

### 获取关卡号
```py
gamestate.level
```
返回类型 ：`int`

表示当前关卡号 

### 获取轮数
```py
gamestate.round
```
返回类型：`int`

表示当前关卡进行到的轮数

### 获取棋盘相关信息
```py
gamestate.board_size
```
返回类型：`int`

表示当前棋盘大小

```py
gamestate.board
```
返回类型：`np.ndarray`

表示棋盘，棋盘中元素 
```
0代表墙
1代表空地
2代表普通豆子
3代表奖励豆子
4代表加速豆子
5代表磁铁豆子
6代表护盾豆子
7代表*2豆子
8代表传送门
```

### 获取吃豆人相关信息
```py
gamestate.pacman_skill_status
```
返回类型：`list[int]`

表示吃豆人当前拥有的技能，数组共4个元素，分别表示<b>DOUBLE_SCORE技能、SPEED_UP技能、MAGNET技能的剩余轮数和当前拥有的SHIELD的数量</b>

```py
gamestate.pacman_pos
```
返回类型：`np.ndarray[int]`

长度为2， `gamestate.pacman_pos[0]` 和 `gamestate.pacman_pos[1]` 分别表示吃豆人的横纵坐标

```
gamestate.pacman_score
```
返回类型：`int`

表示当前吃豆人的得分

### 获取幽灵相关信息
```
gamestate.ghosts_pos
```
返回类型：`np.ndarray[np.ndarray[int]]`

长度为3，表示三个幽灵的坐标

```
gamestate.ghosts_score
```
返回类型：`int`

表示当前幽灵的得分

### 获取传送门相关信息
```
gamestate.portal_available
```
返回类型：`bool`

表示当前传送门是否开启

```
gamestate.portal_coord
```
返回类型：`np.ndarray[int]`

长度为2， `gamestate.portal_coord[0]` 和 `gamestate.portal_coord[1]` 分别表示传送门的横纵坐标

TODO(zyc): 完善除gamestate外其他接口的说明