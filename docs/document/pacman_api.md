# 通信文档

逻辑仓库：[https://github.com/PacMan-Logic/PacmanLogic](https://github.com/PacMan-Logic/PacmanLogic)

## 通信流程
1. judger 在启动游戏逻辑后，向游戏逻辑发送初始化信息。
2. judger 和游戏逻辑开始交互，交互类型分为以下几种：
- 游戏逻辑向 judger 发送各种消息，judger 需要进行解析与处理；
- 游戏逻辑请求 judger 直接将消息原封不动地转发给AI；
- judger 将 AI 发送的消息封装后转发给游戏逻辑；
- judger 将 AI 的异常情况报告给游戏逻辑。
3. 游戏逻辑向 judger 提供玩家得分，并报告游戏结束。在游戏逻辑发送结束消息之前，请务必确保所有操作（尤其是 IO 操作，如写入回放文件）已经完成。

## 报文格式
由于游戏逻辑与 judger 之间使用标准输入输出进行通信，约定好一个标准的报文格式以区分报文的开头和结尾非常重要。

### judger 向逻辑发包：4 + n 格式
数据包的包头为4个字节表示的数据包长度n，字节序为大端序，随后发送数据包正文。

### 逻辑向 judger 发包：4 + 4 + n 格式
数据包的包头由8个字节组成，前4个字节为大端序表示的数据包正文长度，后4个字节为大端序表示的发送目标。当发送目标为-1的时候表示该消息需要 judger 进行解析，当发送目标不为-1的时候表示要转发给的 AI 的编号。

8个字节的包头发送完成后，随后发送数据包正文。

## 通信协议
### 用户->逻辑（正文部分）：json转化为的字符串

#### 作为卷王
```json
{
    "role": 0,
    "action": "action"
}
```

#### 作为幽灵
```json
{
    "role": 1,
    "action": "action1 action2 action3"
}
```
action 为 0/1/2/3/4 分别表示 不动/上/左/下/右

### 逻辑->用户（正文部分）

<b>后端逻辑中的一轮（round）对应与judger通信中的三个回合（state）</b>

第1回合（state）发送座位信息：向0号玩家发送字符串```"0"```，向1号玩家发送字符串```"1"```

每轮（round）开始时，若棋盘改变（进入到新的回合）则向0号玩家和1号玩家发送初始化信息：
```py
{
    "level": self._level,
    "round": self._round,
    "board_size": self._size,
    "board": self._board,
    "pacman_skill_status": np.array(self._pacman.get_skills_status()),
    "pacman_coord": self._pacman.get_coord(), # 卷王坐标
    "ghosts_coord": [ghost.get_coord() for ghost in self._ghosts], # 幽灵坐标
    "score": [self._pacman_score, self._ghosts_score],
    "beannumber": self._beannumber,
    "portal_available": self._portal_available,
    "portal_coord": self._portal_coord,
}
```

每一轮（round）分为三个阶段，对应三个回合（state）

阶段一：读入0号玩家发送的消息，并给1号玩家发送信息```"player 0 send info"```

阶段二：读入1号玩家发送的消息，并给0号玩家发送信息```"player 1 send info"```

阶段三：调用step函数，执行操作，更新局面信息

每局结束发给ai的信息为操作信息
```py
{
    "pacman_action" : pacman.action[0], # 一个数，为卷王的操作
    "ghosts_action" : ghosts.action # 一个含三个元素的数组，为三个幽灵的操作
}
```

每局结束发给播放器的信息为增量信息
```py
{
    "round": self._round, # 当前回合的轮数
    "level": self._level, # 当前回合数
    "pacman_step_block": self._pacman_step_block, # 卷王走过的路径
    "pacman_coord": self._pacman.get_coord(), # 卷王坐标
    "pacman_skills": self._last_skill_status, # 卷王技能
    "ghosts_step_block": self._ghosts_step_block, # 幽灵走过的路径
    "ghosts_coord": [ghost.get_coord() for ghost in self._ghosts], # 幽灵坐标
    "score": [self._pacman_score, self._ghosts_score], # 卷王和幽灵的得分
    "events": [i.value for i in self._event_list], # 事件
    "portal_available": self._portal_available, # 传送门是否已经开启
    "StopReason": None,
}
```
event_list中数与事件的对应如下：
```py
class Event(enum.Enum):
    # 0 and 1 should not occur simutaneously
    EATEN_BY_GHOST = 0 # when eaten by ghost, there are two events to be rendered. first, there should be a animation of pacman being caught by ghost. then, the game should pause for a while, and display a respawning animaiton after receiving next coord infomation.
    SHEILD_DESTROYED = 1 
    # 2 and 3 should not occur simutaneously
    FINISH_LEVEL= 2
    TIMEOUT = 3
```

游戏结束时，会再发送一次信息，其格式与增量信息的格式相同，不过"StopReason"对应的值不为空。