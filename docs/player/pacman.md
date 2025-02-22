# 播放器使用说明
> 2025 Pacman 播放器开发组
## 基本元素
### UI界面
- 左上角:(Debug Mode)进入调试模式
    - 显示当前等级，回合数，吃豆人及幽灵当前坐标及当前回合行动路线
- 页面顶端：显示当前所持有的技能
    - 护学之盾：抵消一次伤害，下方数字为当前剩余护盾数
    - 智引磁石：吸引附近3x3范围内的道具和得分豆，下方数字为磁铁剩余步数
    - 疾风之翼：增加移动速度至`2`，下方数字为加速剩余步数
    - 智慧圣典: 当前回合得分翻倍，下方数字为加倍剩余步数
- 界面最顶端: 显示当前双方得分
- 按"F"可以进行主人物的切换

### 镜头
- 按住鼠标左键可以拖动镜头，使用鼠标滚轮可以进行缩放。
- 在切换地图大小时，镜头会自动回到初始位置并调整到合适的大小。
- 按下"Q"进入全局显示模式，此时镜头会自动回到初始位置
- 按下"W"进行主人物追踪模式
- 按下"E""R""T"分别进入三个幽灵视角追踪模式
- 追踪模式下不能拖动镜头，但可以通过滚轮调整镜头大小
### 游戏元素
- 主人物:
![](./img/main.png)
- 幽灵:
![](./img/ghost1.png)   ![](./img/ghost2.png)   ![](./img/ghost3.png)
- 知识金币:
![](./img/point.png)
- 双倍荣耀币:
![](./img/bonus.png)
- 疾风之翼:
![](./img/acc.png)
- 智引磁石:
![](./img/magnet.png)
- 护学之盾:
![](./img/sh.png)
- 智慧圣典:
![](./img/x2.png)
- 传送门:(初始为关闭状态 , 1/8轮后转变为开启状态)
![](./img/tele_close.png)  ![](./img/tele_open.png)

## 基本操作
### 回放模式
- 使用下方的按钮可以控制回放的相关参数。
    - `<`：回到上一回合
    - `>`：前进到下一回合
    - 播放按钮：开始/暂停自动播放
    - `x2`：调整回放速度
### 交互模式
- 在进行交互模式创建房间时，请注意0号位需要是主人物(AI或玩家)，1号位需要是幽灵(AI或玩家)
- 使用键盘上的方向键 `WASD`进行互动，使用`Enter`进行确认
    - `W`：向上移动
    - `A`：向左移动
    - `S`：向下移动
    - `D`：向右移动
    - `Enter`：在按下方向键后确认移动
- 每次按下方向键都会用红色显示移动后的位置
- 在进行幽灵的移动时，可以使用'1''2''3'进行幽灵的切换，位移操作确定后需要按下回车，其余操作与主人物类似。当前选中的幽灵头顶会有三角形进行标记。只有当三个幽灵的操作均确认后游戏才会继续。
### 说明
如果本次操作不合法（如撞墙），那么吃豆人不会移动