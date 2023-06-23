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


def alfa(t):
    return 0

def beta(t):
    return 0


def onda(f, g, a, b, v, h, k, alfa = alfa, beta = beta):
    # % f es la condici´on inicial de la posici´on
    # % g es la condici´on inicial de la velocidad
    # % v es la velocidad de propagaci´on de la onda
    # % a es la longitud de la cuerda
    # % b es la tiempo que se necesita para evaluar la onda
    # % h es la tama˜no de paso para el espacio
    # % k es la tama˜no de paso para el tiempo
    # % U es la matriz donde se almacena la solucion numerica# Calcular el número de pasos espaciales y temporales
    
    x = np.arange(0, a+h, h)
    t = np.arange(0, b+k, k)
    
    nx = len(x)
    nt = len(t)
    
    # Crear una matriz para almacenar la solución numérica
    U = np.zeros((nx, nt))
    print("Solo ceros")
    imprimir_matriz(U,x,t)
    
    # r es el cálculo para la condición de estabilidad
    r = v * k / h
    
    # condicion inicial
    U[:,0] = f(x)
    
    print("Condiciones Iniciales")
    imprimir_matriz(U,x,t)
    
    # Condiciones de frontera
    U[0, :] = alfa(t)
    U[-1, :] = beta(t)
    
    print("COndiciones de frontera")
    imprimir_matriz(U,x,t)
    
    # Cálculo de las segunda  filas
    for i in range(1, nx - 1):
        print(f"i: {i}")
        U[i, 1] = 2*(1 - r**2) * f(h * (i - 1)) + k * g(h * (i - 1)) + r**2/2.0 * (f(h * i) + f(h * (i - 2)))

    print("Calculo de la segunda fila")
    imprimir_matriz(U,x,t)

    # Cálculo a partir de la tercera fila
    for j in range(1, nt - 1):
        for i in range(1, nx - 1):
            U[i, j+1] = 2*(1-r**2) * U[i, j] + r**2 * (U[i+1, j] + U[i - 1, j]) - U[i, j-1]
            print(f"i : {i} j : {j+1}")
            imprimir_matriz(U,x,t)

    # # Crear las matrices de espacio y tiempo para la gráfica 3D
    # espacio = np.linspace(0, a, num_pasos_espacio)
    # tiempo = np.linspace(0, b, num_pasos_tiempo)
    # espacio, tiempo = np.meshgrid(espacio, tiempo)

    # # Crear la figura 3D y mostrar la solución
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(espacio, tiempo, U, cmap='viridis')
    # ax.set_xlabel('Espacio')
    # ax.set_ylabel('Tiempo')
    # ax.set_zlabel('Amplitud')
    # plt.show()

    # return U
    # imprimir_solucion(U, x, t)
    print("Matriz calculada")
    imprimir_matriz(U,x,t)
    # Crear malla de puntos para el gráfico 3D
    X, T = np.meshgrid(t, x)

    # Crear figura 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Graficar la solución
    ax.plot_surface(T, X, U, cmap='viridis')

    # Configurar etiquetas de los ejes
    ax.set_xlabel('Posicion')
    ax.set_ylabel('Tiempo')
    ax.set_zlabel('Temperatura')

    # Mostrar el gráfico 3D
    # plt.show()

       
# Definir las condiciones iniciales
def f(x):
    return x**2 - x + np.sin(np.pi * x * 2)

def g(x):
    # return np.sin(np.pi * x)
    return 0

# Parámetros
v = 2
a = 1
b = 1
h = 0.3
k = 0.1

onda(f,g,a,b,v,h,k)
