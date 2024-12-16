# 吃豆人SDK引导
本文档是吃豆人和幽灵的SDK使用说明

## 代码结构

SDK由GYM环境类、AI函数和游戏控制器组成。

### GYM环境类
TODO(lxy): git clone以后配置core的说明，不会问zyc
GYM环境类维护了游戏局面的全量信息。该类提供如下接口：

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

### AI函数
一个输入为GYM环境类实例，输出为操作序号的函数。**注意，请不要在函数中直接对传入的环境实例作修改。请使用deepcopy复制**

### 选手可获取的内容及获取方式

#### 获取关卡号
`gamestate.level` 返回一个int值，表示关卡号

`gamestate.round` 返回一个int值，表示当前关卡进行到的轮数

`gamestate.board_size` 返回一个int值，表示当前棋盘大小

`gamestate.board` 返回一个nparray，表示棋盘

棋盘中元素 
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
`gamestate.pacman_skill_status` 返回list[int]，表示吃豆人当前拥有的技能

`gamestate.pacman_pos` 返回list[int]，长度为2，分别为吃豆人的横纵坐标

`gamestate.ghosts_pos` 返回list[list[int]]，长度为3，表示三个幽灵的坐标

`gamestate.pacman_score` 返回一个int值，表示当前吃豆人得分

`gamestate.ghosts_score` 返回一个int值，表示当前幽灵得分

TODO(wxt): 完善gamestate接口说明

TODO(zyc): 完善除gamestate外其他接口的说明