import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def imprimir_matriz(u, x, t, mensaje):    
    nx = len(x)
    nt = len(t)
    # Voltear y rotar la matriz correctamente
    matriz_rotada = np.rot90(u, k=1)

    print(mensaje)

    # Imprimir la matriz con etiquetas en los ejes y colores
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


# Condicinoes de frontera
def alfa(t):
    return 0

def beta(t):
    return 0
  
# Definir las condiciones iniciales
def f(x):
    return x**2 - x + np.sin(np.pi * x * 2)
    # return np.sin(x)

def g(x):
    return np.sin(np.pi * x)
    # return 0

# Ecuacion de la onda
def onda(f, g, long_onda, tiempo_evaluar, velocidad, h, k, condicion_frontera_inicio = alfa, condicion_frontera_final = beta, log = False):
    # f : condicion inicial : posicion
    # g : condicion inicial : velocidad
    # velocidad : velocidad de propagacion de la onda
    # long_onda : longitud de la onda
    # timepo_evaluar : tiempo para evaluar
    # h tamanio division para el espacio
    # k tamanio division para el tiempo
    # U matriz solucion
    
    x = np.arange(0, long_onda+h, h)
    t = np.arange(0, tiempo_evaluar+k, k)
    
    nx = len(x)
    nt = len(t)
    
    # Crear una matriz para almacenar la solución numérica
    U = np.zeros((nx, nt))
    if(log):
        imprimir_matriz(U,x,t,"Solo ceros")
    
    # r es el cálculo para la condición de estabilidad
    r = velocidad * k / h
    
    if(r > 1):
        print("Warning: R inestable")
    
    # condicion inicial
    U[:,0] = f(x)
    
    
    if(log):
        imprimir_matriz(U,x,t, "Condiciones Iniciales")
    
    # Condiciones de frontera
    U[0, :] = condicion_frontera_inicio(t)
    U[-1, :] = condicion_frontera_final(t)
    
    if(log):
        imprimir_matriz(U,x,t, "Condiciones de frontera")
    
    # Cálculo de las segunda  filas
    for i in range(1, nx - 1):
        U[i, 1] = 2*(1 - r**2) * f(h * (i - 1)) + k * g(h * (i - 1)) + r**2/2.0 * (f(h * i) + f(h * (i - 2)))

    if(log):
        imprimir_matriz(U,x,t, "Calculo de la segunda fila")

    # Cálculo a partir de la tercera fila
    for j in range(1, nt - 1):
        for i in range(1, nx - 1):
            U[i, j+1] = 2*(1-r**2) * U[i, j] + r**2 * (U[i+1, j] + U[i - 1, j]) - U[i, j-1]
           
            if(log):
                imprimir_matriz(U,x,t,f"i : {i} j : {j+1}")

    if(log):
        imprimir_matriz(U,x,t, "Matriz Calculada")
        
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

    #Mostrar el gráfico 3D
    plt.show()

def ejemplo_con_log():
    # Parámetros
    v = 2
    a = 1
    b = 1
    h = 0.3
    k = 0.1

    onda(f,g,a,b,v,h,k, log=True)


def ejemplo():
    # Parámetros
    v = 2
    a = 1
    b = 1
    h = 0.03
    k = 0.01

    onda(f,g,a,b,v,h,k)

ejemplo()