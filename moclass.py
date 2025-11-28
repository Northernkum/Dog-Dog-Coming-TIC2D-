import cv2
import numpy as np
import sys
import time
# 移除 YOLOv5 相关导入（无需类别检测）
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import SportClient

# ===================== 关键：定义4个独立的手势检测函数（返回True/False）=====================
# 注意：以下是示例占位函数，需替换为你的实际检测逻辑（比如MediaPipe手势检测、自定义阈值判断等）
def detect_ok_gesture(image):
    """检测OK手势：返回True=检测到，False=未检测到"""
    # TODO：替换为你的OK手势检测逻辑（比如之前的MediaPipe OK检测）
    # 示例：临时返回False，实际使用时需修改
    return False

def detect_palm_gesture(image):
    """检测手掌手势：返回True=检测到，False=未检测到"""
    # TODO：替换为你的手掌手势检测逻辑
    return False

def detect_left7_gesture(image):
    """检测左七字手势：返回True=检测到，False=未检测到"""
    # TODO：替换为你的左七字手势检测逻辑
    return False

def detect_right7_gesture(image):
    """检测右七字手势：返回True=检测到，False=未检测到"""
    # TODO：替换为你的右七字手势检测逻辑
    return False

# ===================== 保留原有的动作配置和状态管理 =====================
# 动作配置（前进/后退速度均为0.2 m/s）
ACTION_CONFIG = {
    "ok": {"action": "前进", "type": "continuous", "params": (0.2, 0, 0)},  # 持续动作
    "palm": {"action": "后退0.4米", "type": "once", "params": (-0.2, 0, 0)},  # 单次动作
    "left7": {"action": "左转90度", "type": "once", "params": (0, 0, 0.4)},    # 单次动作
    "right7": {"action": "右转90度", "type": "once", "params": (0, 0, -0.4)}  # 单次动作
}

# 动作状态管理（避免重复触发单次动作）
ACTION_STATE = {
    "is_executing": False,  # 是否正在执行单次动作（后退/转向）
    "current_action": None, # 当前执行的动作名称
    "start_time": 0         # 动作开始时间
}

# 动作参数校准
BACKWARD_DISTANCE = 0.4  # 目标后退距离（米）
BACKWARD_SPEED = 0.2     # 后退速度（0.2 m/s）
BACKWARD_DURATION = BACKWARD_DISTANCE / abs(BACKWARD_SPEED)  # 后退所需时间（2秒）

TURN_ANGLE = 90          # 目标转向角度（度）
TURN_SPEED = 0.4         # 转向速度（弧度/秒）
TURN_DURATION = (TURN_ANGLE * np.pi / 180) / abs(TURN_SPEED)  # 转向所需时间

# ===================== 保留原有的动作执行函数 =====================
def execute_continuous_action(sport_client, params):
    """执行持续动作（前进）"""
    forward, lateral, turn = params
    sport_client.Move(forward, lateral, turn)
    return f"持续执行：{ACTION_CONFIG['ok']['action']}（速度{forward}m/s）"

def execute_once_action(action_name, params, duration, sport_client):
    """执行单次动作（后退/左转/右转）"""
    global ACTION_STATE
    current_time = time.time()
    
    # 动作未开始→启动动作
    if not ACTION_STATE["is_executing"]:
        ACTION_STATE["is_executing"] = True
        ACTION_STATE["current_action"] = action_name
        ACTION_STATE["start_time"] = current_time
        forward, lateral, turn = params
        sport_client.Move(forward, lateral, turn)
        return f"启动单次动作：{ACTION_CONFIG[action_name]['action']}（预计{duration:.1f}秒）"
    
    # 动作执行中→检查是否完成
    elif ACTION_STATE["current_action"] == action_name:
        elapsed_time = current_time - ACTION_STATE["start_time"]
        if elapsed_time < duration:
            return f"执行中：{ACTION_CONFIG[action_name]['action']}（已运行{elapsed_time:.1f}/{duration:.1f}秒）"
        else:
            # 动作完成→停止
            sport_client.StandDown()
            ACTION_STATE["is_executing"] = False
            ACTION_STATE["current_action"] = None
            return f"单次动作完成：{ACTION_CONFIG[action_name]['action']}（已停止）"
    
    # 正在执行其他单次动作→忽略当前触发
    else:
        return f"正在执行{ACTION_CONFIG[ACTION_STATE['current_action']]['action']}，忽略当前手势"

# ===================== 新增：基于布尔值的手势控制函数 =====================
def gesture_control_by_bool(sport_client, ok_detected, palm_detected, left7_detected, right7_detected):
    """根据4个手势的布尔检测结果，执行对应动作（处理优先级）"""
    global ACTION_STATE
    
    # 优先级：单次动作（后退/转向）> 持续动作（前进）> 未检测到手势（停止）
    action_triggered = False
    
    # 1. 处理单次动作（后退/左转/右转）：仅当无动作执行时触发
    if not ACTION_STATE["is_executing"]:
        # 手掌→后退（单次）
        if palm_detected and not action_triggered:
            control_msg = execute_once_action("palm", ACTION_CONFIG["palm"]["params"], BACKWARD_DURATION, sport_client)
            action_triggered = True
        # 左七字→左转（单次）
        elif left7_detected and not action_triggered:
            control_msg = execute_once_action("left7", ACTION_CONFIG["left7"]["params"], TURN_DURATION, sport_client)
            action_triggered = True
        # 右七字→右转（单次）
        elif right7_detected and not action_triggered:
            control_msg = execute_once_action("right7", ACTION_CONFIG["right7"]["params"], TURN_DURATION, sport_client)
            action_triggered = True
        # OK→前进（持续）
        elif ok_detected and not action_triggered:
            control_msg = execute_continuous_action(sport_client, ACTION_CONFIG["ok"]["params"])
            action_triggered = True
    
    # 2. 正在执行单次动作→继续完成当前动作
    elif ACTION_STATE["is_executing"]:
        current_action = ACTION_STATE["current_action"]
        if current_action == "palm":
            control_msg = execute_once_action(current_action, ACTION_CONFIG[current_action]["params"], BACKWARD_DURATION, sport_client)
        elif current_action in ["left7", "right7"]:
            control_msg = execute_once_action(current_action, ACTION_CONFIG[current_action]["params"], TURN_DURATION, sport_client)
    
    # 3. 未检测到任何手势→停止（执行中动作不受影响）
    else:
        if not ACTION_STATE["is_executing"]:
            sport_client.StandDown()
        control_msg = "未检测到手势→停止（执行中动作不受影响）"
    
    return control_msg

# ===================== 主函数（修改检测逻辑和控制逻辑） =====================
def main():
    # 通信初始化
    ChannelFactoryInitialize(0, sys.argv[1] if len(sys.argv) > 1 else None)
    
    # 摄像头初始化
    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()
    
    # 运动控制初始化
    sport = SportClient()  
    sport.SetTimeout(10.0)
    sport.Init()

    print("=== 布尔型手势控制机器人启动（速度统一0.2m/s）===")
    print("手势映射（检测到返回True即触发）：")
    for action_name, config in ACTION_CONFIG.items():
        speed = config["params"][0] if config["type"] == "continuous" else config["params"][0] if config["params"][0] != 0 else f"{TURN_SPEED}rad/s"
        print(f"  {action_name}手势 → {config['action']}（速度：{speed}）")
    print(f"\n参数校准：后退{BACKWARD_DISTANCE}米（预计{BACKWARD_DURATION:.1f}秒）")
    print(f"         转向90度（预计{TURN_DURATION:.1f}秒）")
    print("按ESC键退出程序")
    
    try:
        while True:
            # 1. 获取摄像头图像
            code, data = client.GetImageSample()
            image_data = np.frombuffer(bytes(data), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            
            if image is None:
                print("无法获取图像，跳过此帧")
                continue
            
            # 2. 调用4个独立的手势检测函数（核心修改：获取布尔结果）
            ok_detected = detect_ok_gesture(image)
            palm_detected = detect_palm_gesture(image)
            left7_detected = detect_left7_gesture(image)
            right7_detected = detect_right7_gesture(image)
            
            # 3. 基于布尔结果执行手势控制
            control_msg = gesture_control_by_bool(sport, ok_detected, palm_detected, left7_detected, right7_detected)
            
            # 4. 可视化显示（修改为显示布尔检测状态）
            # 显示各手势检测状态
            status_text = f"OK: {'检测到' if ok_detected else '未检测到'} | " \
                          f"手掌: {'检测到' if palm_detected else '未检测到'} | " \
                          f"左七字: {'检测到' if left7_detected else '未检测到'} | " \
                          f"右七字: {'检测到' if right7_detected else '未检测到'}"
            cv2.putText(image, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 显示控制状态
            cv2.putText(image, f"控制状态：{control_msg}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示动作进度（若有单次动作执行）
            if ACTION_STATE["is_executing"]:
                current_action = ACTION_STATE["current_action"]
                duration = BACKWARD_DURATION if current_action == "palm" else TURN_DURATION
                progress = (time.time() - ACTION_STATE["start_time"]) / duration
                progress = min(progress, 1.0)
                cv2.putText(image, f"动作进度：{progress*100:.0f}%", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 显示操作说明
            info = "按ESC键退出 | 单次动作执行中忽略其他手势"
            cv2.putText(image, info, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # 显示图像
            cv2.imshow('Gesture Control (Bool Trigger)', image)
            
            # 退出检查
            if cv2.waitKey(1) == 27: 
                print("收到退出信号")
                break
                
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        # 安全关闭
        print("正在安全关闭...")
        sport.StandDown()
        time.sleep(2)
        cv2.destroyAllWindows()
        print("程序已安全退出")

if __name__ == "__main__":
    main()