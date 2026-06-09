#!/usr/bin/env python3
from sympy import *
import matplotlib.pyplot as plt
import math

class Robot:
    def __init__(self, l: tuple = (0.15, 0.30, 0.45)):
        self.l = l
        l1, l2, l3 = l
        self.L = l2 + l3          # eslabón efectivo del antebrazo+EF

        # Rango alcanzable en z=0
        self._R_min = self.L - l1  # 0.60 m
        self._R_max = self.L + l1  # 0.90 m

        # ── Polinomio quíntico lambda (simbólico) ──────────────────────────
        t = symbols("t")
        a0, a1, a2, a3, a4, a5 = symbols("a_0:6")
        lam         = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
        lam_dot     = diff(lam, t)
        lam_dot_dot = diff(lam_dot, t)

        self.t   = t
        self._coeffs  = (a0, a1, a2, a3, a4, a5)
        self.lam          = lam
        self.lam_dot      = lam_dot
        self.lam_dot_dot  = lam_dot_dot

    # ──────────────────────────────────────────────────────────────────────
    # Cinemática directa  →  (x, y, z) del EF
    # ──────────────────────────────────────────────────────────────────────
    def fwd_kin(self, th1, th2, th3):
        # Medidas reales de tu URDF
        h = 0.15       # Altura de la base
        L1 = 0.30      # arm_link
        L2 = 0.45      # forearm_link

        # 1. Deshacer el ajuste del URDF para usar matemática clásica
        th2_math = (math.pi / 2) - th2
        th3_math = -th3

        # 2. Calcular la posición en el plano vertical del brazo
        r = L1 * math.cos(th2_math) + L2 * math.cos(th2_math + th3_math)
        z_rel = L1 * math.sin(th2_math) + L2 * math.sin(th2_math + th3_math)

        # 3. Proyectar al espacio 3D real
        x = r * math.cos(th1)
        y = r * math.sin(th1)
        z = z_rel + h

        return x, y, z

    # ──────────────────────────────────────────────────────────────────────
    # Cinemática inversa para z_ef = 0
    # Entrada : x, y  (punto en plano XY)
    #           _z    (ignorado, compatibilidad con el publisher del profe)
    # Salida  : (th1, th2, th3)
    # ──────────────────────────────────────────────────────────────────────
    def inv_kin(self, x, y, _z=0.0):
        # Medidas reales según tu URDF
        h = 0.15       # Altura de la base
        L1 = 0.30      # arm_link
        L2 = 0.45      # forearm_link

        x = float(x);  y = float(y)
        
        # 1. Ángulo de la base (Yaw)
        th1 = math.atan2(y, x)

        # 2. Distancia en el plano XY (R)
        r = math.hypot(x, y)
        
        # Compensar la altura a la que quieres que llegue (z=0 relativo al suelo)
        z_rel = _z - h 

        # 3. Saturación de seguridad (distancia máxima alcanzable)
        alcance_max = L1 + L2 - 0.01
        dist_objetivo = math.hypot(r, z_rel)
        if dist_objetivo > alcance_max:
            factor = alcance_max / dist_objetivo
            r *= factor
            z_rel *= factor

        # 4. Cinemática inversa (Geometría del triángulo)
        cos_th3 = (r**2 + z_rel**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
        cos_th3 = max(-1.0, min(1.0, cos_th3))
        
        # th3 (codo abajo)
        th3_math = -math.acos(cos_th3)

        k1 = L1 + L2 * math.cos(th3_math)
        k2 = L2 * math.sin(th3_math)
        th2_math = math.atan2(z_rel, r) - math.atan2(k2, k1)

        # 5. AJUSTE DE URDF (Vertical a Horizontal)
        # Como el URDF asume 0 rad = apuntar hacia arriba en Z,
        # necesitamos restar pi/2 (90 grados) para que cuadre con la matemática.
        th2_urdf = (math.pi / 2) - th2_math
        th3_urdf = -th3_math

        return th1, th2_urdf, th3_urdf
    # ──────────────────────────────────────────────────────────────────────
    # Planificación de trayectoria quintica en el plano XY
    # xi_f = (x_f, y_f, beta) — beta se ignora, compatible con ps_callback
    # ──────────────────────────────────────────────────────────────────────
    def def_tray(self,
                 t_f:  float = 2.0,
                 frec: float = 15.0,
                 th_i: tuple = (0.0, 1.0, -1.8),
                 xi_f: tuple = (0.75, 0.0, 0.0)):

        # Posición inicial del EF desde las juntas iniciales
        x_i, y_i, _ = self.fwd_kin(*th_i)
        xi_i = Matrix([x_i, y_i, th_i[0]])   # [x, y, th1_inicial]

        # Posición final
        x_f, y_f = float(xi_f[0]), float(xi_f[1])
        th1_f    = math.atan2(y_f, x_f)
        xi_f_m   = Matrix([x_f, y_f, th1_f])

        # Muestreo
        self.dt       = 1.0 / frec
        self.muestras = int(t_f * frec) + 1

        # ── Coeficientes del polinomio quíntico ───────────────────────────
        t   = self.t
        a0, a1, a2, a3, a4, a5 = self._coeffs
        lam = self.lam

        eq1 = lam.subs(t, 0)
        eq2 = lam.subs(t, t_f) - 1
        eq3 = self.lam_dot.subs(t, 0)
        eq4 = self.lam_dot.subs(t, t_f)
        eq5 = self.lam_dot_dot.subs(t, 0)
        eq6 = self.lam_dot_dot.subs(t, t_f)
        sol = solve([eq1, eq2, eq3, eq4, eq5, eq6],
                    [a0, a1, a2, a3, a4, a5])

        lam_s         = lam.subs(sol)
        lam_dot_s     = self.lam_dot.subs(sol)
        lam_dot_dot_s = self.lam_dot_dot.subs(sol)

        # ── Ecuaciones de trayectoria del EF ─────────────────────────────
        xi_eq         = xi_i + (xi_f_m - xi_i) * lam_s
        xi_dot_eq     = (xi_f_m - xi_i) * lam_dot_s
        xi_dot_dot_eq = (xi_f_m - xi_i) * lam_dot_dot_s

        # ── Muestreo ─────────────────────────────────────────────────────
        t_vals = [self.dt * i for i in range(self.muestras)]

        xi_m         = Matrix.zeros(3, self.muestras)
        xi_dot_m     = Matrix.zeros(3, self.muestras)
        xi_dot_dot_m = Matrix.zeros(3, self.muestras)
        for i, ti in enumerate(t_vals):
            xi_m[:, i]         = xi_eq.subs(t, ti)
            xi_dot_m[:, i]     = xi_dot_eq.subs(t, ti)
            xi_dot_dot_m[:, i] = xi_dot_dot_eq.subs(t, ti)

        # ── Cinemática inversa punto a punto ─────────────────────────────
        th_m = Matrix.zeros(3, self.muestras)
        for i in range(self.muestras):
            th1v, th2v, th3v = self.inv_kin(float(xi_m[0, i]),
                                             float(xi_m[1, i]))
            th_m[0, i] = th1v
            th_m[1, i] = th2v
            th_m[2, i] = th3v

        # ── Derivadas numéricas ───────────────────────────────────────────
        th_dot_m = Matrix.zeros(3, self.muestras)
        if self.muestras > 1:
            th_dot_m[:, 0] = (th_m[:, 1] - th_m[:, 0]) / self.dt
            for i in range(1, self.muestras - 1):
                th_dot_m[:, i] = (th_m[:, i+1] - th_m[:, i-1]) / (2*self.dt)
            th_dot_m[:, -1] = (th_m[:, -1] - th_m[:, -2]) / self.dt

        th_dot_dot_m = Matrix.zeros(3, self.muestras)
        if self.muestras > 2:
            for i in range(1, self.muestras - 1):
                th_dot_dot_m[:, i] = ((th_m[:, i+1] - 2*th_m[:, i] + th_m[:, i-1])
                                      / self.dt**2)
            th_dot_dot_m[:, 0]  = (th_dot_m[:, 1]  - th_dot_m[:, 0])  / self.dt
            th_dot_dot_m[:, -1] = (th_dot_m[:, -1] - th_dot_m[:, -2]) / self.dt

        # ── Guardar ───────────────────────────────────────────────────────
        self.t_arr        = t_vals
        self.xi_m         = xi_m
        self.xi_dot_m     = xi_dot_m
        self.xi_dot_dot_m = xi_dot_dot_m
        self.th_m         = th_m
        self.th_dot_m     = th_dot_m
        self.th_dot_dot_m = th_dot_dot_m

    # ──────────────────────────────────────────────────────────────────────
    # Graficación  (misma interfaz que el robot del profe)
    # ──────────────────────────────────────────────────────────────────────
    def _plot3(self, title, labels, data):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle(title)
        for ax, lbl, col, row in zip(axes, labels,
                                      ["red", "green", "blue"],
                                      range(3)):
            ax.set_title(lbl)
            ax.plot(self.t_arr,
                    [float(data[row, i]) for i in range(self.muestras)],
                    color=col)
            ax.set_xlabel("t (s)")
        plt.tight_layout()
        plt.show()

    def imp_tray(self):
        self._plot3("Postura del EF (plano XY)",
                    ["x (m)", "y (m)", "th1=beta (rad)"],
                    self.xi_m)

    def imp_vel(self):
        self._plot3("Velocidades del EF",
                    ["x_dot", "y_dot", "th1_dot"],
                    self.xi_dot_m)

    def imp_acc(self):
        self._plot3("Aceleraciones del EF",
                    ["x_ddot", "y_ddot", "th1_ddot"],
                    self.xi_dot_dot_m)

    def imp_junt(self):
        self._plot3("Posiciones de juntas",
                    ["th1 (shoulder yaw)", "th2 (arm pitch)", "th3 (forearm pitch)"],
                    self.th_m)

    def imp_junt_vel(self):
        self._plot3("Velocidades de juntas",
                    ["th1_dot", "th2_dot", "th3_dot"],
                    self.th_dot_m)

    def imp_junt_acc(self):
        self._plot3("Aceleraciones de juntas",
                    ["th1_ddot", "th2_ddot", "th3_ddot"],
                    self.th_dot_dot_m)


# ──────────────────────────────────────────────────────────────────────────
# Main de prueba independiente (sin ROS)
# ──────────────────────────────────────────────────────────────────────────
def main():
    robot = Robot()

    # Configuración inicial: th1=0, th2=acos(valor), th3 correspondiente
    th1_i = 0.0
    th2_i = math.acos(max(-1.0, min(1.0,
                (0.75**2 + 0.15**2 - 0.75**2) / (2*0.75*0.15))))   # R=0.75
    sin_phi = -(0.15/0.75)*math.sin(th2_i)
    cos_phi = (0.75 - 0.15*math.cos(th2_i))/0.75
    phi_i   = math.atan2(sin_phi, cos_phi)
    th3_i   = phi_i - th2_i

    robot.def_tray(
        t_f=3.0,
        frec=15,
        th_i=(th1_i, th2_i, th3_i),
        xi_f=(0.5, 0.5, 0.0)          # click en (0.5, 0.5) del plano XY
    )

    robot.imp_tray()
    robot.imp_junt()
    robot.imp_junt_vel()
    robot.imp_junt_acc()


if __name__ == "__main__":
    main()