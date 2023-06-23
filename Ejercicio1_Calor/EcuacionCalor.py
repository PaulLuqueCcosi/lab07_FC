import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def imprimir_matriz(u, x, t, mensaje):
    nx = len(x)
    nt = len(t)

    # Voltear y rotar la matriz correctamente
    matriz_rotada = np.rot90(u, k=1)

    # Imprimir la matriz con etiquetas en los ejes y colores
    print(mensaje)
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
        
def ecuacion_calor(f, long_alambre, tiempo_evaluar, coe_difusion, h, k,condicion_frontera_inicial=alfa, condicion_frontera_final=beta, log= False):
    # f=u(x,0) es la condición inicial
    # condicion_frontera_inicio =u(0,t)
    # condicion_frontera_final=u(a,t)
    # long_alambre:  es el ancho del alambre
    # tiempo_evaluar :  es el tiempo de la simulacion
    # coe_difusion: coeficiente de difusion
    # h es el tamaño de paso en el espacio
    # k es el tamaño de paso en el tiempo
    
    x = np.arange(0, long_alambre + h, h)
    t = np.arange(0, tiempo_evaluar + k, k)
    nx = len(x)
    nt = len(t)

    # Matriz para almacenar los valores de la solución
    u = np.zeros((nx, nt))

    # Condiciones iniciales
    u[:, 0] = f(x)
    if(log):
        imprimir_matriz(u,x,t, "Condiciones Iniciales")

    # Condiciones en la frontera
    u[0, :] = condicion_frontera_inicial(t)
    u[-1, :] = condicion_frontera_final(t)
    if(log):
        imprimir_matriz(u,x,t, "Condiciones de frontera")

    # definimos
    r = coe_difusion**2*k/h**2
    
    if(r<0 or r > 0.5):
        print(f"Warning: r {r} puede ser inestable")
        print(f"con v: {coe_difusion}, h : {h}")
        condicion = h**2/2*coe_difusion**2
        print(f"k < {condicion}")
        exit()
    
    # Iteración para calcular la solución
    for j in range(1, nt):
        for i in range(1, nx - 1):
            u[i, j] = (1-2*r)*u[i,j-1] + r*(u[i+1,j-1] + u[i-1, j-1])
            if(log):
                imprimir_matriz(u,x,t,f"i : {i} j : {j}")

    
    if(log):
        imprimir_matriz(u,x,t, "Matriz Calculada")
        
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

def ejemplo():
    longitud_alambre = 4
    tiempo = 1.5
    coeficiente_difusion = 1
    h = 0.05
    k = 0.0012

    ecuacion_calor(f, longitud_alambre, tiempo, coeficiente_difusion, h, k)
    
def ejemplo_con_log():
    longitud_alambre = 5
    tiempo = 0.2
    coeficiente_difusion = 1
    h = 0.5
    k = 0.125

    ecuacion_calor(f, longitud_alambre, tiempo, coeficiente_difusion, h, k, log=True)
    
ejemplo_con_log()