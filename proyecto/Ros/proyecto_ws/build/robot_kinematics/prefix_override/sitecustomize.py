import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/robotics/Downloads/CursoRobotica-Edu/proyecto/install/robot_kinematics'
