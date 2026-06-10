import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/robotics/Robotica2026-2/proyecto/Ros/proyecto_ws/src/install/robot_hardware'
