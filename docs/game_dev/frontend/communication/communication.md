## 补充文件

<a href="/game_dev/frontend/communication/index.html" download>index.html</a>
<a href="/game_dev/frontend/communication/player.html" download>player.html</a>

这两个文件是最后build之后，将其放置在根目录下，修改一些常数后，一起打包部署上saiblo的

player.html负责和播放器进行通信操作

## 接受网页信息

前端需要负责的通信为前端和html的通信，大体的逻辑是：

场景中需要有一个叫做`Main Controller`的组件，html会调用unity实例中这个组件的某些函数，当然组件名和函数名也可以改，不过同时需要修改html文件

![img](/imgs/communication/image1.png)

上图为`player.html`第85行开始的片段，这段代码位于一个`window.addEventListener`中，代表为当前窗口添加了一个事件监听函数，

- 第一个参数为`message`代表这是监听网页收到的信息，而这部分信息是由judger传过来的
- 第二个参数是一个回调函数，其中定义了将后端发过来的信息传递给播放器的方法`send_to_player`，其中第89行是核心逻辑，`SendMessage`的三个参数分别是：
  - 需要通信的组件名称
  - 需要调用的这个组件下的函数名（注意不要同名！）
  - 调用的参数

阅读后面的代码可以发现，发过来的信息会通过这个`send_to_player`发送给播放器，因此我们需要：

- 定义一个`Main Controller`组件
- 下方需要挂载一个脚本，这个脚本中需要有一个名为`HandleMessage`的函数
- 这个函数接受一个字符串作为参数

这样的话播放器就可以**接受网页信息**了，接受网页信息后，根据官方文档的协议对数据做相应的处理即可

## 向网页发送信息

向网页发送信息需要用到一些奇怪的函数，位于消消乐仓库的`/Assets/Plugins/Output.jslib`中，这是一个祖传的JS库函数，其中四个函数的作用分别是：

- `Send_frontend`: 向网页发消息
- `Connect_ws`: 与Judger建立websocket连接
- `Send_ws`: 向websocket发送信息
- `Getoperation`: 在回放模式的时候，这个函数会用来处理回放文件每一行

这些函数的最终实现都是在`player.html`中

具体到消消乐中，我们向网页发送信息的函数如下图，而`FrontendReplyData`是规定好的向网页发送信息时的格式，具体的格式可以参照官方文档或代码中的定义

![img](/imgs/communication/image2.png)

![img](/imgs/communication/image3.png)

注意到，我们与网页通信实际上只会在初始化阶段进行，即游戏开始之后就不需要网页通信了，我们会通过websocket直接与Judger进行通信，因此这些函数都不应该在逻辑中被调用，所以设置为了`private`

## 与后端通信

与后端的通信是通过Judger来进行的，也即unity -- Judger -- backend，我们与Judger的通信方式是通过websocket进行的，这就需要我们首先建立一个ws连接，建立的方式是：

首先网页会首先向播放器发送一个在线模式初始化信息：

![img](/imgs/communication/image4.png)

我们在上述`HandleMessage`函数中对这个`json`格式进行处理，得到`token`之后，根据这个`token`请求建立ws连接，建立的方式为如下，这个函数为解析`token`字符串，并相应的处理对应部分，详细`token`的组成规则请参阅官方文档

![img](/imgs/communication/image5.png)

建立成功后我们便可以利用这个ws来与Judger进行通信，而事实上webGL是不支持ws连接的，这个时候就说明了我们为什么需要html套壳——我们要利用window的ws连接与后端进行通信，而调用window的ws连接的方式是通过`Output.jslib`文件中定义的函数

具体来说，我们与后端通信的函数为下面两个函数，具体的通信协议需要和后端商议（建议尽早确定，并且提升通信协议的可扩展性和鲁棒性）

![img](/imgs/communication/image6.png)

## 附注

在消消乐中，与网络通信相关的脚本都定义在`WebInteractionController`中，并且给出了比较详细的注释，大家可以参考

## 通信协议 

### 网页->Unity的信息

网页会两种情况下向玩家发送信息：

- 初始化阶段
- 离线模式（看回放）的时候用户拖动进度条、调倍速等

格式如下，如果只在某个模式下使用会特殊标注：

```JSON
{
    "message": xx; // 消息类型，可能的值为下表中MsgType所示
    "payload": xx; // 离线模式，回放文件总帧数
    "token": xx; // 在线模式，用于后续解析后与judger连接的token
    "speed": xx; // 离线模式，用户调整倍速时，对应的倍速
    "replay_data": xx; // 离线模式初始化，传过来的回放文件，不需要我们解析！！！！
    "index": xx; // 离线模式下，网页向unity请求播放特定帧，对应的帧数
    "players": xx; // 参与游戏的玩家名称列表
}
public enum MsgType
{
    init_player_player, // 在线模式初始化
    init_replay_player, // 回放模式初始化
    load_frame, // 加载某一帧（index为帧数）
    load_next_frame, // 加载下一帧
    load_players, // 加载玩家名称
    play_speed, // 修改回放速度
}
```

### Unity->网页信息

```JSON
{
    "message": xx; // 消息类型，可能的值为下表中MsgType所示
    "number_of_frames": xx; // 离线模式，告知回放文件总帧数
    "height": xx; // 告知播放器高度 不需要操作
    "init_result": xx; // 离线模式，回放文件是否解析成功
    "game_record": xx; // 没用过，不知道啥
    "err_msg": xx; // 错误信息
}
public enum MsgType
{
    init_successfully,
    initialize_result,
    game_record,    
    error_marker,
    loaded // 表示当前unity已经初始化完成，可以开始接收评测机的信息
}
```

## 回放模式解析文件的流程

![img](/imgs/communication/image7.png)

这是官网的文档，从这个话来看，我们会拿到一个Blob对象，然后手动解析，但是我们看消消乐的代码会发现，replay_data这个玩意根本没有被使用过，所以是为什么呢？我们来分析一下

### 为什么要发loaded

还是来看`player.html`中的这段代码，我们之前只关注了`if`分支中的过程，现在我们需要来看一下`else`分支，进入`else`分支的条件是，网页上这个unity播放器还没被创建或还未加载好（这个很好理解）

![img](/imgs/communication/image8.png)

在`else`分支中，我们会将本该传递给用户的信息`__payload`存在一个列表`src_msg`里面，接下来就是怎么处理`src_msg`的问题了，很直接的一个想法是，我们需要将这些信息在加载好之后再传过去，而`Saiblo`本身也是这么做的：

![img](/imgs/communication/image9.png)

这一段代码是定义了window的一个函数`handlePlayerCalls`，从名字上来看这就是在处理用户发过来的信息，而我们根据上述Unity->网页的通信协议可以知道，其中有一个`message`属性用于判断消息类型，在这里就是处理`message='loaded'`的情况，这里首先让`src_msg`做了一个reverse操作，也就是相当于把这玩意当成一个队列来处理，按照FIFO(First In First Out)的方法将缓存下来的信息依次传递给用户，同样的是通过`HandleMessage`来处理

那这个`handlePlayerCalls`函数在哪调的呢？对没错`Output.jslib`中的`Send_frontend`

### 回放文件的解析

好说了这么多，我们现在理解了为什么要给他发一个`loaded`了，也就是说我们会存在一些信息，这些信息是在unity甚至都不存在的时候就会发给网页`player.html`了，那这部分信息是啥呢

就是**解析好了的回放文件**

![img](/imgs/communication/image10.png)

在网页收到初始化离线模式信息的时候，并不会直接发送给播放器，而是新建一个`FileReader`对象，对这个`replay_data`进行读取、解析与存储，从图中107-109行可以看出，最后发给前端播放器的信息中，`replay_data`是一个空数组，而`payload`恰恰就是回放文件的帧数！（这里要`-1`我也没太想明白，可能是最后有一个空行之类的？）

暂停一下，到这一步，完成的工作时，网页端解析了回放文件，存在一个叫`line_array`的数组中，并且把帧数封装成信息传给了用户（也可能是进入了`src_msg`中）

下一步，就是我们什么时候`line_array`中的内容传递给用户

![img](/imgs/communication/image11.png)

`line_array`传递给用户是在这个函数里进行的，网页把用户指定的项传递过去，而unity显然不能直接调用js的函数，这个时候就又需要`Output.jslib`出场了，我们可以通过其中定义的`Getoperation`函数来调用这个`SendOperation`函数，进而在unity中实现`HandleOperation`函数，来实现对回放文件每一行的详细处理

![img](/imgs/communication/image12.png)

上图是消消乐中处理前端发过来的初始化回放模式的完整逻辑，可以看出我们在得到帧数之后，遍历调用`Getoperation`函数来拿到对应的信息，而`HandleOperation`实现的就是对回放文件每一行的处理：

![img](/imgs/communication/image13.png)

### 总流程

1. 网页端收到回放文件消息，对其进行解析、处理、储存，将`init_replay_player`传递给unity（可能会堵塞）
2. unity加载完成，向网页端发送`loaded`
3. 网页端接收到loaded，将缓存的消息队列`src_img`依次发送给unity
4. unity接收到`init_replay_player`，开始准备从网页获取回放文件
5. unity调用`Getoperation`，进而调用`window.SendOperation`，这个函数将特定帧的回放信息交给`HandleOperation`处理
6. 在`HandleOperation`实现处理逻辑

### 附注

这么写的原因是，去年Generals的回放文件太大，导致传不过来，于是文物朱志杭就改了`player.html`，为了代码的鲁棒性，我们决定就这么传递下去

## 本地子进程测试

由于Unity输出的WebGL不具备websocket通信功能，在saiblo正式服和测试服中使用html代替WebGL进行websocket通信。但是由于在正式服和测试服上部署以及输出WebGL都需要一段时间，不适合大批量的测试和修改。因此建议在使用html代替播放器通信前，在本地启动子进程，游戏逻辑以及Judger来测试播放器与逻辑的通信协议以及解析。

### 测试方式

使用本地子进程测试，需要在本地准备[judger.py](https://git.tsinghua.edu.cn/hgr18/saiblo-judge/-/blob/master/interactive_judger/judger.py?ref_type=heads)，游戏逻辑程序，打开播放器的Unity编辑器以及一个输出在设备上运行的播放器（或一个AI）。在播放器中，使用Websocket直接发送预计在saiblo上发送的信息，将URL设置为本地地址和特定端口（注意编辑器中播放器的端口和输出播放器中的端口应该不同0）。

### 测试命令

```Bash
python [path of judger.py] test_mode
0 [player_index] [command to run the AI] #启动AI对战
1 [player_index] 127.0.0.1 [port] [room_id] #对应播放器中的URL为：127.0.0.1:[port]/[room_id]/0/[player_index]
4 [command to run the logic] 0 [path to replay file] #输入这条命令前，需要播放器连接，连接成功judger会有提示
```

对于启动命令的更详细信息，可以参考[judger的使用文档](https://git.tsinghua.edu.cn/hgr18/saiblo-judge/-/tree/master/interactive_judger?ref_type=heads)（学长的传承）
