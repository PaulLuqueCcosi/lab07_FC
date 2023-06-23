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






        
def heat_equation(f, alfa, beta, a, b, c, h, k):
    # f=u(x,0) es la condición inicial
    # alfa=u(0,t) y beta=u(a,t) condiciones en la frontera
    # a es el ancho del alambre
    # b es el tiempo de la simulacion
    # c coeficiente de difusion
    # h es el tamaño de paso en el espacio
    # k es el tamaño de paso en el tiempo
    
    x = np.arange(0, a + h, h)
    t = np.arange(0, b + k, k)
    nx = len(x)
    nt = len(t)

    # Matriz para almacenar los valores de la solución
    u = np.zeros((nx, nt))

    # Condiciones iniciales
    u[:, 0] = f(x)

    # Condiciones en la frontera
    u[0, :] = alfa(t)
    u[-1, :] = beta(t)

    # Iteración para calcular la solución
    for j in range(1, nt):
        for i in range(1, nx - 1):
            u[i, j] = u[i, j-1] + c * k / h**2 * (u[i + 1, j-1] - 2 * u[i, j-1] + u[i - 1, j-1])

    imprimir_solucion(u, x, t)
    
    imprimir_matriz(u,x,t)
    # Crear malla de puntos para el gráfico 3D
    X, T = np.meshgrid(t, x)

    # Crear figura 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Graficar la solución
    ax.plot_surface(T, X, u, cmap='viridis')

    # Configurar etiquetas de los ejes
    ax.set_xlabel('Posicion')
    ax.set_ylabel('Tiempo')
    ax.set_zlabel('Temperatura')

    # Mostrar el gráfico 3D
    plt.show()

# Ejemplo de uso
def f(x):
    return np.sin(np.pi/2 * x)

def alfa(t):
    return 0

def beta(t):
    return 0

a = 2
b = 0.2
c = 1
h = 0.5
k = 0.2

heat_equation(f, alfa, beta, a, b, c, h, k)