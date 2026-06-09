#!/usr/bin/env python3
"""
Robot RRR — cinemática directa, inversa y planificación de trayectoria.
Dimensiones por defecto: h=0.20 m (shoulder), L1=0.35 m (arm), L2=0.50 m (forearm).
"""
from sympy import symbols, diff, solve, Matrix
import matplotlib.pyplot as plt
import math


class Robot:
    """
    Modelo cinemático de un brazo RRR (3 DOF).

    Parámetros
    ----------
    l : tuple(h, L1, L2)
        h  — longitud del primer eslabón (shoulder / altura de la base).
        L1 — longitud del segundo eslabón (arm).
        L2 — longitud del tercer eslabón (forearm + EF).
    """

    def __init__(self, l: tuple = (0.20, 0.35, 0.50)):
        self.l = l
        self.h, self.L1, self.L2 = l   # fuente única de verdad para las dimensiones

        self._R_min = self.L1 + self.L2 - self.h   # alcance mínimo en z=0
        self._R_max = self.L1 + self.L2 + self.h   # alcance máximo en z=0

        # Polinomio quíntico simbólico — se reutiliza en _quintic_profile
        t = symbols("t")
        a0, a1, a2, a3, a4, a5 = symbols("a_0:6")
        lam         = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
        lam_dot     = diff(lam, t)
        lam_dot_dot = diff(lam_dot, t)

        self.t           = t
        self._coeffs     = (a0, a1, a2, a3, a4, a5)
        self.lam         = lam
        self.lam_dot     = lam_dot
        self.lam_dot_dot = lam_dot_dot

    # ---- Cinemática directa ------------------------------------------------

    def fwd_kin(self, th1, th2, th3):
        """(th1, th2, th3) → (x, y, z) del extremo efector."""
        h, L1, L2 = self.h, self.L1, self.L2

        # Ajuste convención URDF (0 rad = vertical) → matemática clásica
        th2_m = (math.pi / 2) - th2
        th3_m = -th3

        r     = L1 * math.cos(th2_m) + L2 * math.cos(th2_m + th3_m)
        z_rel = L1 * math.sin(th2_m) + L2 * math.sin(th2_m + th3_m)

        x = r * math.cos(th1)
        y = r * math.sin(th1)
        z = z_rel + h

        return x, y, z

    # ---- Cinemática inversa ------------------------------------------------

    def inv_kin(self, x, y, _z=0.0):
        """
        (x, y[, _z]) → (th1, th2, th3).
        _z se acepta por compatibilidad con el publisher del profesor.
        """
        h, L1, L2 = self.h, self.L1, self.L2
        x, y = float(x), float(y)

        # 1. Ángulo de la base (Yaw)
        th1   = math.atan2(y, x)
        r     = math.hypot(x, y)
        z_rel = _z - h

        # 2. Saturación al alcance máximo
        alcance_max   = L1 + L2 - 0.01
        dist_objetivo = math.hypot(r, z_rel)
        if dist_objetivo > alcance_max:
            f      = alcance_max / dist_objetivo
            r     *= f
            z_rel *= f

        # 3. Cinemática inversa — geometría del triángulo (codo abajo)
        cos_th3 = (r**2 + z_rel**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
        cos_th3 = max(-1.0, min(1.0, cos_th3))
        th3_m   = -math.acos(cos_th3)

        k1    = L1 + L2 * math.cos(th3_m)
        k2    = L2 * math.sin(th3_m)
        th2_m = math.atan2(z_rel, r) - math.atan2(k2, k1)

        # 4. Ajuste matemática → convención URDF
        th2_urdf = (math.pi / 2) - th2_m
        th3_urdf = -th3_m

        return th1, th2_urdf, th3_urdf

    # ---- Helpers de planificación -----------------------------------------

    def _quintic_profile(self, t_f):
        """Resuelve el polinomio quíntico y retorna (lam_s, lam_dot_s, lam_ddot_s)."""
        t   = self.t
        a0, a1, a2, a3, a4, a5 = self._coeffs
        lam = self.lam

        sol = solve([
            lam.subs(t, 0),
            lam.subs(t, t_f) - 1,
            self.lam_dot.subs(t, 0),
            self.lam_dot.subs(t, t_f),
            self.lam_dot_dot.subs(t, 0),
            self.lam_dot_dot.subs(t, t_f),
        ], [a0, a1, a2, a3, a4, a5])

        return (lam.subs(sol),
                self.lam_dot.subs(sol),
                self.lam_dot_dot.subs(sol))

    def _central_diff(self, mat):
        """Diferencias finitas centrales (con diferencia hacia adelante/atrás en bordes)."""
        N   = mat.cols
        dt  = self.dt
        out = Matrix.zeros(3, N)
        if N > 1:
            out[:, 0]  = (mat[:, 1]  - mat[:, 0])  / dt
            for i in range(1, N - 1):
                out[:, i] = (mat[:, i+1] - mat[:, i-1]) / (2 * dt)
            out[:, -1] = (mat[:, -1] - mat[:, -2]) / dt
        return out

    # ---- Planificación de trayectoria -------------------------------------

    def def_tray(self,
                 t_f:  float = 2.0,
                 frec: float = 15.0,
                 th_i: tuple = (0.0, 1.0, -1.8),
                 xi_f: tuple = (0.75, 0.0, 0.0)):

        # Posición inicial y final del EF
        x_i, y_i, _ = self.fwd_kin(*th_i)
        xi_i   = Matrix([x_i, y_i, th_i[0]])
        x_f, y_f = float(xi_f[0]), float(xi_f[1])
        xi_f_m = Matrix([x_f, y_f, math.atan2(y_f, x_f)])

        # Muestreo
        self.dt       = 1.0 / frec
        self.muestras = int(t_f * frec) + 1
        t_vals        = [self.dt * i for i in range(self.muestras)]

        # Perfil quíntico y ecuaciones de trayectoria del EF
        lam_s, lam_dot_s, lam_ddot_s = self._quintic_profile(t_f)
        delta         = xi_f_m - xi_i
        xi_eq         = xi_i + delta * lam_s
        xi_dot_eq     = delta * lam_dot_s
        xi_dot_dot_eq = delta * lam_ddot_s

        # Muestreo del EF
        t = self.t
        xi_m         = Matrix.zeros(3, self.muestras)
        xi_dot_m     = Matrix.zeros(3, self.muestras)
        xi_dot_dot_m = Matrix.zeros(3, self.muestras)
        for i, ti in enumerate(t_vals):
            xi_m[:, i]         = xi_eq.subs(t, ti)
            xi_dot_m[:, i]     = xi_dot_eq.subs(t, ti)
            xi_dot_dot_m[:, i] = xi_dot_dot_eq.subs(t, ti)

        # Cinemática inversa punto a punto
        th_m = Matrix.zeros(3, self.muestras)
        for i in range(self.muestras):
            th1v, th2v, th3v = self.inv_kin(float(xi_m[0, i]), float(xi_m[1, i]))
            th_m[0, i] = th1v
            th_m[1, i] = th2v
            th_m[2, i] = th3v

        # Derivadas numéricas de las juntas
        th_dot_m     = self._central_diff(th_m)
        th_dot_dot_m = self._central_diff(th_dot_m)

        # Guardar resultados
        self.t_arr        = t_vals
        self.xi_m         = xi_m
        self.xi_dot_m     = xi_dot_m
        self.xi_dot_dot_m = xi_dot_dot_m
        self.th_m         = th_m
        self.th_dot_m     = th_dot_m
        self.th_dot_dot_m = th_dot_dot_m

    # ---- Graficación -------------------------------------------------------

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


# ---- Main de prueba (sin ROS) ---------------------------------------------

def main():
    robot = Robot()   # l=(0.20, 0.35, 0.50)

    th1_i = 0.0
    th2_i = math.acos(max(-1.0, min(1.0,
                (0.75**2 + 0.20**2 - 0.75**2) / (2 * 0.75 * 0.20))))
    sin_phi = -(0.20 / 0.75) * math.sin(th2_i)
    cos_phi = (0.75 - 0.20 * math.cos(th2_i)) / 0.75
    phi_i   = math.atan2(sin_phi, cos_phi)
    th3_i   = phi_i - th2_i

    robot.def_tray(
        t_f=3.0,
        frec=15,
        th_i=(th1_i, th2_i, th3_i),
        xi_f=(0.5, 0.5, 0.0),
    )

    robot.imp_tray()
    robot.imp_junt()
    robot.imp_junt_vel()
    robot.imp_junt_acc()


if __name__ == "__main__":
    main()
