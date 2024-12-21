# 热身赛SDK引导
本文档是热身赛（开心消消乐）的SDK使用说明

## 代码结构

SDK由GYM环境类、AI函数和游戏控制器组成。

### GYM环境类

GYM环境类维护了游戏局面的全量信息。该类提供如下接口：

#### reset
- Args
    - seed (int, optional): 环境随机数生成器的随机种子。和操作共同完全决定局面的转移. Defaults to None.
    - board (numpy.ndarray, optional): 初始棋盘。一般不使用. Defaults to None.

- Returns:
    - numpy.ndarray: 返回初始化后的棋盘

#### step
- Args:
    - action (int): 操作的序号。可以由coord_to_num方法从坐标转换得到。操作序号的定义见下文。
    - player (int, optional): 进行操作的玩家序号. Defaults to 0.

- Returns:
    - tuple[np.ndarray, int, bool]: 分别为局面、当次操作的reward以及游戏是否结束

#### observe
无参数。根据render_mode决定输出。本地使用时将render_mode设置为`'local'`，在调用后可以在命令行打印当前局面。

#### num_to_coord 和coord_to_num
分别为将操作序号转换为坐标，和将坐标转换为操作序号。
操作坐标的定义为：`[x1, y1, x2, y2]`代表交换坐标`[x1, y1]`和`[x2, y2]`的块。
操作序号定义为 $y_2 + x_2 \times size + y_1 \times size^2 + x_1 \times size^3$


### Controller
在初始化时读取judger传来的初始化信息并实例化整局游戏唯一的环境实例。run将调用传入的ai函数，并将ai函数返回的操作序号传给judger，并读取对方操作。

### AI函数
一个输入为GYM环境类实例，输出为操作序号的函数。**注意，请不要在函数中直接对传入的环境实例作修改。请使用deepcopy复制**