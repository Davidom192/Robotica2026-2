# Proyecto Final de Robótica

¡Bienvenidos al repositorio de nuestro proyecto final de robótica! En este proyecto, desarrollamos, simulamos y controlamos un sistema robótico utilizando herramientas y estándares de la industria actual.

---

## Team
* **Dana Anzaldo**
* **Fernando Carrizosa**
* **Eduardo Jimenez**

---

## 🛠️ Tecnologías y Herramientas Utilizadas

El proyecto fue desarrollado y probado utilizando el siguiente ecosistema de software:

* **Sistema Operativo:** Ubuntu (Linux)
* **Framework de Robótica:** ROS 2 (Robot Operating System)
* **Modelado del Robot:** URDF (Unified Robot Description Format)
* **Visualización y Simulación:** RViz2
* **Entorno de Desarrollo (IDE):** Visual Studio Code (VS Code)
* **Control de Versiones:** Git & GitHub

---

## 📂 Estructura del Proyecto

A continuación se muestra la estructura principal del espacio de trabajo (*workspace*) de ROS 2:

```text
mi_proyecto_robotica/
├── src/
│   └── mi_robot_description/
│       ├── CMakeLists.txt
│       ├── package.xml
│       ├── urdf/
│       │   └── mi_robot.urdf.xacro   # Modelo geométrico del robot
│       ├── launch/
│       │   └── display.launch.py     # Script para lanzar RViz2 y cargar el URDF
│       └── rviz/
│           └── config.rviz           # Configuración guardada de RViz
└── README.md