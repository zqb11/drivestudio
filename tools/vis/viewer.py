from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
from jaxtyping import Float32, UInt8
from nerfview import CameraState, Viewer
from viser import Icon, ViserServer

from tools.vis.playback_panel import add_gui_playback_group


# 封装了viser服务器的交互式3D查看类
class DynamicViewer(Viewer):# 继承Viewer
    def __init__(
        self,
        server: ViserServer,
        render_fn: Callable[
            [CameraState, Tuple[int, int]],
            Union[
                UInt8[np.ndarray, "H W 3"],
                Tuple[UInt8[np.ndarray, "H W 3"], Optional[Float32[np.ndarray, "H W"]]],
            ],
        ],
        num_frames: int,
        #work_dir: str,
        mode: Literal["rendering", "training"] = "rendering",
    ):
        self.num_frames = num_frames
        # self.work_dir = Path(work_dir)
        super().__init__(server, render_fn, mode)# 调用父类Viewer的构造函数初始化属性
        # self._define_guis()# 创建交互式gui控件

    def _define_guis(self):
        #super()._define_guis()
        server = self.server# viser服务端提供gui控件的创建接口
        self._time_folder = server.gui.add_folder("Time")# 创建Time栏，用于放置时间相关组件
        with self._time_folder:# 在Time栏中创建组件
            self._playback_guis = add_gui_playback_group(
                server,
                num_frames=self.num_frames,# 总帧数
                initial_fps=15.0,# 初始播放帧率
            )# 添加一组播放控件，返回包含播放控件的列表
            self._playback_guis[0].on_update(self.rerender)# 为播放控件绑定更新事件，当播放控件状态改变时触发rerender函数重新渲染场景