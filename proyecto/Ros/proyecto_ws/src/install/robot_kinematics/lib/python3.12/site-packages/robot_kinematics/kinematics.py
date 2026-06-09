#!/usr/bin/env python3
from sympy import *
import matplotlib.pyplot as plt
import math

class Robot():
  def __init__(self):

    th1, th2, th3 = symbols("theta_1 theta_2 theta_3")

    # geometría
    self.L1 = 0.30
    self.L2 = 0.45
    self.h = 0.15
    self.l = (self.L1, self.L2, 0.0)

    # cinemática directa
    r = self.L1*cos(th2) + self.L2*cos(th2 + th3)

    x = r*cos(th1)
    y = r*sin(th1)
    z = self.h + self.L1*sin(th2) + self.L2*sin(th2 + th3)

    self.xi_0_p = Matrix([x, y, z])

    # Jacobiano correcto
    self.J = self.xi_0_p.jacobian([th1, th2, th3])

    # trayectoria
    t = symbols("t")

    a = symbols("a_0:6")
    self.a_0, self.a_1, self.a_2, self.a_3, self.a_4, self.a_5 = a

    self.lam = sum(a[i]*t**i for i in range(6))
    self.lam_dot = diff(self.lam, t)
    self.lam_dot_dot = diff(self.lam_dot, t)

    self.t = t

    self.th1 = th1
    self.th2 = th2
    self.th3 = th3

  def def_tray(self, t_f:float=2, frec:float=15, 
               th_i:tuple[float]=(0.1, 0.1,0.1), 
               xi_f:tuple[float]=(0.6, 0.1, 0)):
    
    # Posición del efector final substituyendo en la postura inicial (m, rad)
    xi_i = self.xi_0_p.subs({self.th1: th_i[0], 
                             self.th2: th_i[1], 
                             self.th3: th_i[2]})
    # Muestreo y dt
    self.dt = 1.0/frec
    self.muestras = int(t_f * frec) + 1

    # Eq. de restricción para trayectoria
    eq1 = self.lam.subs({self.t: 0})
    eq2 = self.lam.subs({self.t: t_f}) - 1
    eq3 = self.lam_dot.subs({self.t: 0})
    eq4 = self.lam_dot.subs({self.t: t_f})
    eq5 = self.lam_dot_dot.subs({self.t: 0})
    eq6 = self.lam_dot_dot.subs({self.t: t_f})
    solutions = solve((eq1, eq2, eq3, eq4, eq5, eq6),
                  (self.a_0, self.a_1, self.a_2, self.a_3, self.a_4, self.a_5))
    # Sustituyendo solución en polinomio lambda
    lam_s         = self.lam.subs(solutions)
    lam_dot_s     = self.lam_dot.subs(solutions)
    lam_dot_dot_s = self.lam_dot_dot.subs(solutions)
    
    # Ecuación de posiciones, velocidades y aceleración del E.F.
    xi_f = Matrix([xi_f[0], xi_f[1], xi_f[2]])
    xi_eq         = xi_i + (xi_f - xi_i) * lam_s
    xi_dot_eq     = (xi_f - xi_i) * lam_dot_s
    xi_dot_dot_eq = (xi_f - xi_i) * lam_dot_dot_s
    
    # Arreglos para muestreo
    t_m = Matrix.zeros(1, self.muestras)
    for i in range(self.muestras):
      t_m[i] = self.dt * i

    xi_m         = Matrix.zeros(3, self.muestras)
    xi_dot_m     = Matrix.zeros(3, self.muestras)
    xi_dot_dot_m = Matrix.zeros(3, self.muestras)
    for i in range(self.muestras):
      xi_m[:, i]         = xi_eq.subs({self.t: t_m[i]})
      xi_dot_m[:, i]     = xi_dot_eq.subs({self.t: t_m[i]})
      xi_dot_dot_m[:, i] = xi_dot_dot_eq.subs({self.t: t_m[i]})

    # Cinemática inversa por puntos de trayectoria
    th_m = Matrix.zeros(3, self.muestras)
    for i in range(self.muestras):
      th_m[:, i] = self.inv_kin(xi_m[0, i], xi_m[1, i], xi_m[2, i])

    # Derivadas numéricas
    th_dot_m = Matrix.zeros(3, self.muestras)
    if self.muestras > 1:
      th_dot_m[:, 0] = (th_m[:, 1] - th_m[:, 0]) / self.dt
      for i in range(1, self.muestras - 1):
        th_dot_m[:, i] = (th_m[:, i+1] - th_m[:, i-1]) / (2 * self.dt)
      th_dot_m[:, self.muestras - 1] = (th_m[:, self.muestras - 1] - th_m[:, self.muestras - 2]) / self.dt

    th_dot_dot_m = Matrix.zeros(3, self.muestras)
    if self.muestras > 2:
      for i in range(1, self.muestras - 1):
        th_dot_dot_m[:, i] = (th_m[:, i+1] - 2 * th_m[:, i] + th_m[:, i-1]) / (self.dt ** 2)
      th_dot_dot_m[:, 0] = (th_dot_m[:, 1] - th_dot_m[:, 0]) / self.dt
      th_dot_dot_m[:, self.muestras - 1] = (th_dot_m[:, self.muestras - 1] - th_dot_m[:, self.muestras - 2]) / self.dt

    # Guardar variables en la clase
    self.xi_m         = xi_m
    self.xi_dot_m     = xi_dot_m
    self.xi_dot_dot_m = xi_dot_dot_m
    self.th_m         = th_m
    self.th_dot_m     = th_dot_m
    self.th_dot_dot_m = th_dot_dot_m
    self.t_m = t_m

  def inv_kin(self, x, y, z):

    L1, L2, _ = self.l

    x = float(x)
    y = float(y)
    z = float(z)

    # 1. ángulo base
    th1 = math.atan2(y, x)

    # 2. proyección en plano del brazo
    r = math.sqrt(x**2 + y**2)
    z = z - self.h

    # 3. ley del coseno
    cos_t3 = (r**2 + z**2 - L1**2 - L2**2) / (2*L1*L2)
    cos_t3 = max(-1.0, min(1.0, cos_t3))

    th3 = math.acos(cos_t3)

    # 4. codo abajo (evita configuraciones raras)
    if z < 0:
        th3 = -th3

    k1 = L1 + L2*math.cos(th3)
    k2 = L2*math.sin(th3)

    th2 = math.atan2(z, r) - math.atan2(k2, k1)

    return Matrix([th1, th2, th3])
  


  def imp_tray(self):
    fig, (x_g, z_g, be_g) = plt.subplots(nrows = 1, ncols = 3)
    fig.suptitle("Posiciones del efector final")
    x_g.set_title("x")
    z_g.set_title("z")
    be_g.set_title("beta")
    x_g.plot(self.t_m.T,  self.xi_m[0, :].T, color="RED")
    z_g.plot(self.t_m.T,  self.xi_m[1, :].T, color="green")
    be_g.plot(self.t_m.T, self.xi_m[2, :].T, color=(0,0,1))
    plt.show()
    pass
  def imp_junt(self):
    fig, (th1_g, th2_g, th3_g) = plt.subplots(nrows = 1, ncols = 3)
    fig.suptitle("Posiciones de las juntas")
    th1_g.set_title("th1")
    th2_g.set_title("th2")
    th3_g.set_title("th3")
    th1_g.plot(self.t_m.T,  self.th_m[0, :].T, color="RED")
    th2_g.plot(self.t_m.T,  self.th_m[1, :].T, color="green")
    th3_g.plot(self.t_m.T,  self.th_m[2, :].T, color=(0,0,1))
    plt.show()
    pass

  def imp_junt_vel(self):
    fig, (th1_g, th2_g, th3_g) = plt.subplots(nrows = 1, ncols = 3)
    fig.suptitle("Velocidades de las juntas")
    th1_g.set_title("th1_dot")
    th2_g.set_title("th2_dot")
    th3_g.set_title("th3_dot")
    th1_g.plot(self.t_m.T,  self.th_dot_m[0, :].T, color="RED")
    th2_g.plot(self.t_m.T,  self.th_dot_m[1, :].T, color="green")
    th3_g.plot(self.t_m.T,  self.th_dot_m[2, :].T, color=(0,0,1))
    plt.show()
    pass

  def imp_junt_acc(self):
    fig, (th1_g, th2_g, th3_g) = plt.subplots(nrows = 1, ncols = 3)
    fig.suptitle("Aceleraciones de las juntas")
    th1_g.set_title("th1_ddot")
    th2_g.set_title("th2_ddot")
    th3_g.set_title("th3_ddot")
    th1_g.plot(self.t_m.T,  self.th_dot_dot_m[0, :].T, color="RED")
    th2_g.plot(self.t_m.T,  self.th_dot_dot_m[1, :].T, color="green")
    th3_g.plot(self.t_m.T,  self.th_dot_dot_m[2, :].T, color=(0,0,1))
    plt.show()
    pass

  def imp_pos(self):
    fig, (x_g, z_g, be_g) = plt.subplots(nrows = 1, ncols = 3)
    fig.suptitle("Posiciones del efector final")
    x_g.set_title("x")
    z_g.set_title("z")
    be_g.set_title("beta")
    x_g.plot(self.t_m.T,  self.xi_m[0, :].T, color="RED")
    z_g.plot(self.t_m.T,  self.xi_m[1, :].T, color="green")
    be_g.plot(self.t_m.T, self.xi_m[2, :].T, color=(0,0,1))
    plt.show()
    pass

  def imp_vel(self):
    fig, (x_dot_g, z_dot_g, be_dot_g) = plt.subplots(nrows = 1, ncols = 3)
    fig.suptitle("Velocidades del efector final")
    x_dot_g.set_title("x_dot")
    z_dot_g.set_title("z_dot")
    be_dot_g.set_title("beta_dot")
    x_dot_g.plot(self.t_m.T,  self.xi_dot_m[0, :].T, color="RED")
    z_dot_g.plot(self.t_m.T,  self.xi_dot_m[1, :].T, color="green")
    be_dot_g.plot(self.t_m.T, self.xi_dot_m[2, :].T, color=(0,0,1))
    plt.show()
    pass

  def imp_acc(self):
    fig, (x_dot_dot_g, z_dot_dot_g, be_dot_dot_g) = plt.subplots(nrows = 1, ncols = 3)
    fig.suptitle("Aceleraciones del efector final")
    x_dot_dot_g.set_title("x_dot_dot")
    z_dot_dot_g.set_title("z_dot_dot")
    be_dot_dot_g.set_title("beta_dot_dot")
    x_dot_dot_g.plot(self.t_m.T,  self.xi_dot_dot_m[0, :].T, color="RED")
    z_dot_dot_g.plot(self.t_m.T,  self.xi_dot_dot_m[1, :].T, color="green")
    be_dot_dot_g.plot(self.t_m.T, self.xi_dot_dot_m[2, :].T, color=(0,0,1))
    plt.show()
    pass

  def tr_h(self, x=0, y=0, z=0,
                 gamma=0, beta=0, alpha=0):
    t_x = Matrix([[1,          0,           0, x],
                  [0, cos(gamma), -sin(gamma), 0],
                  [0, sin(gamma),  cos(gamma), 0],
                  [0,          0,           0, 1]])
    t_y = Matrix([[ cos(beta),          0, sin(beta), 0],
                  [         0,          1,         0, y],
                  [-sin(beta),          0, cos(beta), 0],
                  [         0,          0,         0, 1]])
    t_z = Matrix([[cos(alpha), -sin(alpha), 0, 0],
                  [sin(alpha),  cos(alpha), 0, 0],
                  [         0,           0, 1, z],
                  [         0,           0, 0, 1]])
    tr = simplify(t_x * t_y * t_z)
    return tr

def main():
  robot = Robot()
  robot.def_tray()
  robot.imp_tray()
  robot.imp_junt()
  robot.imp_junt_vel()
  robot.imp_junt_acc()
  robot.imp_pos()
  robot.imp_vel()
  robot.imp_acc()
if __name__ == "__main__":
  main()
