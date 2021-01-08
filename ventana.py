import matplotlib.pyplot as plt
import numpy as np
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import backpropagation

root = tkinter.Tk()
root.wm_title("Backpropagation")

fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])

canvas = FigureCanvasTkAgg(fig, master=root)
#canvas.draw()
canvas.get_tk_widget().grid(column=0, row=0, padx=5, pady=5, columnspan=10, rowspan=10)

entradas = []
salidas = []
pesos = []


def onclick(event):
    # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
    if claseActiva.get() == 1:
        entradas.append([event.xdata, event.ydata])
        if numeroDeClases.get() == 3:
            salidas.append([1, 0, 0])
        elif numeroDeClases.get() == 4:
            salidas.append([1, 0, 0, 0])
        else:
            salidas.append([1, 0, 0, 0, 0])
        plt.scatter(event.xdata, event.ydata, color="m")
        # for i, valor in enumerate(entradas):
    elif claseActiva.get() == 2:
        entradas.append([event.xdata, event.ydata])
        if numeroDeClases.get() == 3:
            salidas.append([0, 1, 0])
        elif numeroDeClases.get() == 4:
            salidas.append([0, 1, 0, 0])
        else:
            salidas.append([0, 1, 0, 0, 0])
        plt.scatter(event.xdata, event.ydata, color="c")
    elif claseActiva.get() == 3:
        entradas.append([event.xdata, event.ydata])
        plt.scatter(event.xdata, event.ydata, color="b")
        if numeroDeClases.get() == 3:
            salidas.append([0, 0, 1])
        elif numeroDeClases.get() == 4:
            salidas.append([0, 0, 1, 0])
        else:
            salidas.append([0, 0, 1, 0, 0])
    elif claseActiva.get() == 4:
        entradas.append([event.xdata, event.ydata])
        plt.scatter(event.xdata, event.ydata, color="r")
        if numeroDeClases.get() == 4:
            salidas.append([0, 0, 0, 1])
        else:
            salidas.append([0, 0, 0, 1, 0])
    else:
        entradas.append([event.xdata, event.ydata])
        if numeroDeClases.get() == 5:
            salidas.append([0, 0, 0, 0, 1])
        plt.scatter(event.xdata, event.ydata, color="g")
    fig.canvas.draw()


def cerrar():
    root.quit()
    root.destroy()

def numeroClases():
    n = numeroDeClases.get()
    if n == 3:
        claseActiva4.config(state = tkinter.DISABLED)
        claseActiva5.config(state = tkinter.DISABLED)
    elif n == 4:
        claseActiva4.config(state = tkinter.NORMAL)
        claseActiva5.config(state = tkinter.DISABLED)
    else:
        claseActiva4.config(state = tkinter.NORMAL)
        claseActiva5.config(state = tkinter.NORMAL)

def entrenar():
    x = np.array(entradas)
    y = np.array(salidas)
    bp = backpropagation.Backpropagation(
        x,
        y,
        epocas.get(),
        errorDeseado.get(),
        tasaAprendizaje.get(),
        1,
        2,
        numeroDeClases.get(),
    )
    errorCuadratico, numeroDeEpocas, pesos1, pesosSalida = bp.backpropagation()

    fig1, ax1 = plt.subplots()
    ax1.set_ylabel("Error")
    ax1.set_title("Epoca")
    ax1.plot(numeroDeEpocas, errorCuadratico)
    #plot_decision_boundary(x, y, pesos1, pesosSalida)
    fig1.show()


def plot_decision_boundary(X, y, pesos1, pesosSalida, steps=1000, cmap='twilight'):
    cmap = plt.get_cmap(cmap)
    xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
    ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1
    h = 0.1
    l = np.linspace(-5, 5, 100)
    xx, yy = np.meshgrid(l, l)
    fig2, ax = plt.subplots()
    #ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)
    #train_labels = model.predict(X)
    ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, lw=0)

    fig2.show()



tkinter.Label(root, text="Capas Ocultas").grid(column=2, row=11, padx=5, pady=5)
capasOcultas = tkinter.IntVar()
capa1 = tkinter.Radiobutton(root, text="1", variable=capasOcultas, value=1).grid(
    column=2, row=12, padx=5, pady=5
)
capa2 = tkinter.Radiobutton(root, text="2", variable=capasOcultas, value=2).grid(
    column=2, row=13, padx=5, pady=5
)
capasOcultas.set(1)

tkinter.Label(root, text="Epocas:").grid(column=3, row=11, padx=5, pady=5)
epocas = tkinter.IntVar()
spinboxEpocas = tkinter.Spinbox(
    root, bd=3, from_=1000, to=10000, increment=100, textvariable=epocas, width=10
)
spinboxEpocas.grid(column=4, row=11, padx=5, pady=5)

tkinter.Label(root, text="Tasa de aprendizaje:").grid(column=3, row=12, padx=5, pady=5)
tasaAprendizaje = tkinter.DoubleVar()
spinboxTasaAprendizaje = tkinter.Spinbox(
    root,
    bd=3,
    from_=0.10,
    to=0.90,
    increment=0.10,
    textvariable=tasaAprendizaje,
    width=10,
)
spinboxTasaAprendizaje.grid(column=4, row=12, padx=5, pady=5)

tkinter.Label(root, text="Error deseado:").grid(column=3, row=13, padx=5, pady=5)
errorDeseado = tkinter.DoubleVar()
errorDeseado.set(0.50)
spinboxErrorDeseado = tkinter.Spinbox(
    root, bd=3, from_=0.10, to=0.90, increment=0.10, textvariable=errorDeseado, width=10
)
spinboxErrorDeseado.grid(column=4, row=13, padx=5, pady=5)

tkinter.Label(root, text="Numero de clases").grid(column=5, row=11, padx=5, pady=5)
numeroDeClases = tkinter.IntVar()
clases3 = tkinter.Radiobutton(root, text="3", variable=numeroDeClases, value=3).grid(
    column=5, row=12, padx=5, pady=5
)
clases4 = tkinter.Radiobutton(root, text="4", variable=numeroDeClases, value=4).grid(
    column=5, row=13, padx=5, pady=5
)
clases5 = tkinter.Radiobutton(root, text="5", variable=numeroDeClases, value=5).grid(
    column=5, row=14, padx=5, pady=5
)
numeroDeClases.set(3)
buttonClases = tkinter.Button(master=root, text="N. clases", command=numeroClases)
buttonClases.grid(column=5, row=15, padx=5, pady=5)

tkinter.Label(root, text="Clase activa a agregar").grid(
    column=6, row=11, padx=5, pady=5
)
claseActiva = tkinter.IntVar()
claseActiva1 = tkinter.Radiobutton(root, text="1", variable=claseActiva, value=1).grid(
    column=6, row=12, padx=5, pady=5
)
claseActiva2 = tkinter.Radiobutton(root, text="2", variable=claseActiva, value=2).grid(
    column=6, row=13, padx=5, pady=5
)
claseActiva3 = tkinter.Radiobutton(root, text="3", variable=claseActiva, value=3).grid(
    column=6, row=14, padx=5, pady=5
)
claseActiva4 = tkinter.Radiobutton(root, text="4", variable=claseActiva, value=4).grid(
    column=6, row=15, padx=5, pady=5
)
claseActiva5 = tkinter.Radiobutton(root, text="5", variable=claseActiva, value=5).grid(
    column=6, row=16, padx=5, pady=5
)
claseActiva.set(1)

buttonEntrenar = tkinter.Button(master=root, text="Entrenar", command=entrenar)
buttonEntrenar.grid(column=7, row=11, padx=5, pady=5)

buttonCerrar = tkinter.Button(master=root, text="Cerrar", command=cerrar)
buttonCerrar.grid(column=7, row=12, padx=5, pady=5)

cid = fig.canvas.mpl_connect("button_press_event", onclick)
tkinter.mainloop()