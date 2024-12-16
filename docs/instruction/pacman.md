# 吃豆人开发指南
本文档为吃豆人的开发指南。本文档基于热身赛sdk进行开发，sdk的使用说明可以参考引导文档sdk部分的对应文档。

## stage1: 实现一个能运行的ai
### 吃豆人
我们首先实现一个能够在平台上运行的agent。

根据sdk的文档，我们需要实现一个ai函数，该函数接收一个GameState类型的对象，返回一个<b>长度为1</b>的数组，数组元素为`[0, 4]`之间的整数。我们可以使用随机采样实现该函数：
```python
# ai.py
def ai(game_state: GameState):
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

## stage2：实现一个有策略的ai
### 吃豆人
TODO(zyc): 样例ai

### 幽灵
TODO(zyc): 样例ai

## stage3：炼丹

## stage4：可能的改进