#!/usr/bin/env python3
from sympy import *
import matplotlib.pyplot as plt
import math

class Robot():
  def __init__(self, l:tuple[float]=(0.15, 0.30, 0.45)):
    th1, th2, th3 = symbols("theta_1,theta_2,theta_3")

    h = l[0]
    L1 = l[1]
    L2 = l[2]

    # Postura del efector final
    r = L1*cos(th2) + L2*cos(th2 + th3)
    x = r*cos(th1)
    y = r*sin(th1)
    z = h + L1*sin(th2) + L2*sin(th2 + th3)

    xi_0_p = Matrix([x, y, z])

    t = symbols("t")
    a_0, a_1, a_2, a_3, a_4, a_5 = symbols("a_0:6")
    lam = a_0 + a_1*t + a_2*t**2 + a_3*t**3 + a_4*t**4 + a_5*t**5    
    lam_dot = diff(lam, t)
    lam_dot_dot = diff(lam_dot, t)
    
    self.th1, self.th2, self.th3 = th1, th2, th3
    self.xi_0_p = xi_0_p
    self.a_0, self.a_1, self.a_2, self.a_3, self.a_4, self.a_5 = a_0, a_1, a_2, a_3, a_4, a_5
    self.t = t
    self.lam, self.lam_dot, self.lam_dot_dot = lam, lam_dot, lam_dot_dot
    self.l = l

  def inv_kin_exacta(self, x, y, z):
    h = self.l[0]; L1 = self.l[1]; L2 = self.l[2]
    
    th1 = math.atan2(y, x)
    r = math.hypot(x, y)
    z_rel = z - h
    
    # Saturación contra singularidades
    alcance_max = L1 + L2 - 0.001
    dist = math.hypot(r, z_rel)
    if dist > alcance_max:
        r = r * (alcance_max / dist)
        z_rel = z_rel * (alcance_max / dist)
        
    cos_th3 = (r**2 + z_rel**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
    cos_th3 = max(-1.0, min(1.0, cos_th3))
    
    # Codo arriba
    th3_math = -math.acos(cos_th3)
    
    k1 = L1 + L2 * math.cos(th3_math)
    k2 = L2 * math.sin(th3_math)
    th2_math = math.atan2(z_rel, r) - math.atan2(k2, k1)
    
    # Ajuste físico al URDF
    th2_urdf = (math.pi / 2) - th2_math
    th3_urdf = -th3_math
    
    return Matrix([th1, th2_urdf, th3_urdf])

  def def_tray(self, t_f:float=2, frec:float=15, 
               th_i:tuple[float]=(0.1, 0.1, 0.1), 
               xi_f:tuple[float]=(0.6, 0.1, 0.0)):
    
    # 1. TRADUCIR ángulos del URDF de vuelta a la matemática pura
    th1_math = th_i[0]
    th2_math = (math.pi / 2) - th_i[1]
    th3_math = -th_i[2]

    # 2. Calcular la posición inicial real usando los ángulos corregidos
    xi_i = self.xi_0_p.subs({
        self.th1: th1_math, 
        self.th2: th2_math, 
        self.th3: th3_math
    }).evalf() # evalf() asegura que sea un número flotante y no una fracción simbólica
    
    self.dt = 1.0/frec
    self.muestras = int(t_f * frec) + 1 

    # ... (El resto del código hacia abajo se queda exactamente igual)
    eq1 = self.lam.subs({self.t: 0})
    eq2 = self.lam.subs({self.t: t_f}) - 1
    eq3 = self.lam_dot.subs({self.t: 0})
    eq4 = self.lam_dot.subs({self.t: t_f})
    eq5 = self.lam_dot_dot.subs({self.t: 0})
    eq6 = self.lam_dot_dot.subs({self.t: t_f})
    solutions = solve((eq1, eq2, eq3, eq4, eq5, eq6),
                  (self.a_0, self.a_1, self.a_2, self.a_3, self.a_4, self.a_5))
    
    lam_s         = self.lam.subs(solutions)
    lam_dot_s     = self.lam_dot.subs(solutions)
    lam_dot_dot_s = self.lam_dot_dot.subs(solutions)
    
    xi_f_mat = Matrix([xi_f[0], xi_f[1], xi_f[2]])
    xi_eq         = xi_i + (xi_f_mat - xi_i) * lam_s
    xi_dot_eq     = (xi_f_mat - xi_i) * lam_dot_s
    xi_dot_dot_eq = (xi_f_mat - xi_i) * lam_dot_dot_s
    
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
    
    # ---- Cinemática Inversa Analítica Exacta ----
    th_m = Matrix.zeros(3, self.muestras)
    for i in range(self.muestras):
        # En lugar del Jacobiano, calculamos el punto exacto garantizando precisión absoluta
        th_m[:, i] = self.inv_kin_exacta(float(xi_m[0, i]), float(xi_m[1, i]), float(xi_m[2, i]))
    
    # Derivadas numéricas (Diferencias finitas) para completar tu estructura
    th_dot_m     = Matrix.zeros(3, self.muestras)
    th_dot_dot_m = Matrix.zeros(3, self.muestras)
    
    if self.muestras > 1:
        th_dot_m[:, 0] = (th_m[:, 1] - th_m[:, 0]) / self.dt
        for i in range(1, self.muestras - 1):
            th_dot_m[:, i] = (th_m[:, i+1] - th_m[:, i-1]) / (2 * self.dt)
        th_dot_m[:, -1] = (th_m[:, -1] - th_m[:, -2]) / self.dt

    if self.muestras > 2:
        for i in range(1, self.muestras - 1):
            th_dot_dot_m[:, i] = (th_m[:, i+1] - 2*th_m[:, i] + th_m[:, i-1]) / (self.dt**2)
            
    self.xi_m = xi_m; self.xi_dot_m = xi_dot_m; self.xi_dot_dot_m = xi_dot_dot_m
    self.th_m = th_m; self.th_dot_m = th_dot_m; self.th_dot_dot_m = th_dot_dot_m
    self.t_m = t_m

  def imp_tray(self):
    fig, (x_g, y_g, z_g) = plt.subplots(nrows = 1, ncols = 3, figsize=(10, 3))
    fig.suptitle("Posiciones del efector final")
    
    # Extraer valores numéricos
    t_vals = [float(v) for v in self.t_m]
    x_vals = [float(v) for v in self.xi_m[0, :]]
    y_vals = [float(v) for v in self.xi_m[1, :]]
    z_vals = [float(v) for v in self.xi_m[2, :]]

    x_g.set_title("x")
    x_g.plot(t_vals, x_vals, color="red", linewidth=2)
    
    y_g.set_title("y")
    y_g.plot(t_vals, y_vals, color="green", linewidth=2)
    
    z_g.set_title("z")
    z_g.plot(t_vals, z_vals, color="blue", linewidth=2)
    
    # Si la línea es totalmente plana, forzamos los límites para verla
    if max(z_vals) - min(z_vals) < 0.01:
        z_g.set_ylim(-0.1, 0.1)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)

  def imp_junt(self):
    fig, (th1_g, th2_g, th3_g) = plt.subplots(nrows = 1, ncols = 3, figsize=(10, 3))
    fig.suptitle("Posiciones de las juntas")
    
    # Extraer valores numéricos
    t_vals = [float(v) for v in self.t_m]
    th1_vals = [float(v) for v in self.th_m[0, :]]
    th2_vals = [float(v) for v in self.th_m[1, :]]
    th3_vals = [float(v) for v in self.th_m[2, :]]

    th1_g.set_title("th1")
    th1_g.plot(t_vals, th1_vals, color="red", linewidth=2)
    
    th2_g.set_title("th2")
    th2_g.plot(t_vals, th2_vals, color="green", linewidth=2)
    
    th3_g.set_title("th3")
    th3_g.plot(t_vals, th3_vals, color="blue", linewidth=2)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)