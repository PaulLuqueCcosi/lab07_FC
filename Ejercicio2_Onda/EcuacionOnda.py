import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def imprimir_solucion(u, x, t):
    nx = len(x)
    nt = len(t)

    for j in range(nt):
        print(f"t = {t[j]}:")
        for i in range(nx):
            print(f"x = {x[i]} | u(x,t) = {u[i, j]}")

        print()

def imprimir_matriz(u, x, t):
    # % f es la condici´on inicial de la posici´on
    # % g es la condici´on inicial de la velocidad
    # % v es la velocidad de propagaci´on de la onda
    # % a es la longitud de la cuerda
    # % b es la tiempo que se necesita para evaluar la onda
    # % h es la tama˜no de paso para el espacio
    # % k es la tama˜no de paso para el tiempo
    # % U es la matriz donde se almacena la solucion numerica
    nx = len(x)
    nt = len(t)

    # Voltear y rotar la matriz correctamente
    matriz_rotada = np.rot90(u, k=1)

    # Imprimir la matriz con etiquetas en los ejes y colores
    print("Matriz de solución:")
    print('Horizontal : x\nVertical : t\n')

    for j in range(nt):
        print(f"{t[(-1*(j+1))]: <8.2f}", end="")
        for i in range(nx):
            valor = matriz_rotada[j, i]
            color = '\033[92m'
            print(f"{color}{valor: <8.4f}\033[0m", end="")
        print()
    print('\t', end="")
    for i in range(nx):
        print(f"{x[i]: <8.2f}", end="")
    print()




def onda(f, g, a, b, v, h, k):
    # Calcular el número de pasos espaciales y temporales
    num_pasos_espacio = int(a / h) + 1
    num_pasos_tiempo = int(b / k) + 1

    # r es el cálculo para la condición de estabilidad
    r = v * k / h
    r1 = r ** 2
    r2 = r ** 2 / 2
    s1 = 1 - r ** 2
    s2 = 2 * (1 - r ** 2)

    # Crear una matriz para almacenar la solución numérica
    U = np.zeros((num_pasos_tiempo, num_pasos_espacio))

    # Cálculo de las primeras dos filas
    for i in range(1, num_pasos_espacio - 1):
        U[0, i] = f(h * (i - 1))
        U[1, i] = s1 * f(h * (i - 1)) + k * g(h * (i - 1)) + r2 * (f(h * i) + f(h * (i - 2)))

    # Cálculo a partir de la tercera fila
    for j in range(1, num_pasos_tiempo - 1):
        for i in range(1, num_pasos_espacio - 1):
            U[j + 1, i] = s2 * U[j, i] + r1 * (U[j, i - 1] + U[j, i + 1]) - U[j - 1, i]

    # Crear las matrices de espacio y tiempo para la gráfica 3D
    espacio = np.linspace(0, a, num_pasos_espacio)
    tiempo = np.linspace(0, b, num_pasos_tiempo)
    espacio, tiempo = np.meshgrid(espacio, tiempo)

    # Crear la figura 3D y mostrar la solución
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(espacio, tiempo, U, cmap='viridis')
    ax.set_xlabel('Espacio')
    ax.set_ylabel('Tiempo')
    ax.set_zlabel('Amplitud')
    plt.show()

    return U

       
# Definir las condiciones iniciales
def f(x):
    return np.sin(np.pi * x / a)

def g(x):
    return 0

# Parámetros
v = 2
a = 20
b = 20
h = 0.05
k = 0.01

u , espacio, tiempo = onda(f,g,a,b,v,h,k)
print(u)
print(espacio)
print(tiempo)
