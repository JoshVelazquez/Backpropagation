import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from random import seed
from random import random
from datetime import datetime

class Backpropagation:
    def __init__(
        self,
        entradas,
        salidas,
        epocas,
        errorDeseado,
        tasaDeAprendizaje,
        capasOcultas,
        neuronasCapa1,
        numeroDeClases,
    ):
        self.entradas = entradas
        self.salidas = salidas
        self.epocas = epocas
        self.errorDeseado = errorDeseado
        self.tasaDeAprendizaje = tasaDeAprendizaje
        self.capasOcultas = capasOcultas
        self.neuronasCapa1 = neuronasCapa1
        self.numeroDeClases = numeroDeClases
        self.pesos1 = np.zeros((neuronasCapa1, 3))
        self.pesos2 = []
        self.pesosSalida = np.zeros((numeroDeClases, 3))

    def backpropagation(self):
        errorCuadratico = np.finfo(np.float64).max
        arrayErrorCuadratico = []
        arrayEpocas = []
        convergio = False
        epoca = 0
        sensibilidad1 = np.zeros(self.neuronasCapa1)
        sensibilidad3 = np.zeros(self.numeroDeClases)
        self.inicializarPesos()
        while epoca < self.epocas and errorCuadratico > self.errorDeseado:
            errorCuadratico = 0
            error = np.zeros(self.numeroDeClases)
            net1 = np.zeros(self.neuronasCapa1)
            net3 = np.zeros(self.numeroDeClases)
            salida1 = np.zeros(self.neuronasCapa1)
            salida3 = np.zeros(self.numeroDeClases)
            for vectorEntrada, vectorSalida in zip(self.entradas, self.salidas):
                # Forward
                for i in range(self.neuronasCapa1):
                    # temp = self.pesos1[i]
                    net1[i] = self.pesos1[i][0] + np.dot(
                        vectorEntrada, self.pesos1[i][1:]
                    )
                    salida1[i] = self.funcionSigmoide(net1[i])

                for i in range(self.numeroDeClases):
                    net3[i] = self.pesosSalida[i][0] + np.dot(
                        salida1, self.pesosSalida[i][1:]
                    )
                    salida3[i] = self.funcionSigmoide(net3[i])

                # Backward
                for i in range(self.numeroDeClases):
                    resultado = vectorSalida[i] - salida3[i]
                    sensibilidad3[i] = (
                        self.derivadaFuncionSigmoide(net3[i]) * resultado
                    )
                    error[i] = resultado

                transPesos1 = np.transpose(self.pesos1)
                for i in range(self.neuronasCapa1):
                    sensibilidad1[i] = (
                        self.derivadaFuncionSigmoide(net1[i])
                        * transPesos1[0][i]
                        * sensibilidad3[0]
                        + self.derivadaFuncionSigmoide(net1[i])
                        * transPesos1[1][i]
                        * sensibilidad3[1]
                        + self.derivadaFuncionSigmoide(net1[i])
                        * transPesos1[2][i]
                        * sensibilidad3[2]
                    )

                # actualizar pesos
                transEntradas1 = np.transpose(vectorEntrada)
                for neurona in range(self.neuronasCapa1):
                    for i in range(len(vectorEntrada)):
                        self.pesos1[neurona][i + 1] += (
                            self.tasaDeAprendizaje
                            * sensibilidad1[i]
                            * transEntradas1[i]
                        )
                    self.pesos1[neurona][0] += (
                        self.tasaDeAprendizaje * sensibilidad1[neurona]
                    )

                transEntradas3 = np.transpose(salida1)
                for neurona in range(self.numeroDeClases):
                    for i in range(len(salida1)):
                        self.pesosSalida[neurona][i + 1] += (
                            self.tasaDeAprendizaje
                            * sensibilidad3[i]
                            * transEntradas3[i]
                        )
                    self.pesosSalida[neurona][0] += (
                        self.tasaDeAprendizaje * sensibilidad3[neurona]
                    )
                errorCuadratico += sum(error)**2
                
            epoca += 1
            errorCuadratico /= len(self.entradas)
            arrayErrorCuadratico.append(errorCuadratico)
            arrayEpocas.append(epoca)
            if errorCuadratico < self.errorDeseado:
                exp_scores = np.exp(sensibilidad3)
                probs = exp_scores / np.sum(exp_scores, keepdims=True)
                print("Convergio uwu en la epoca: " + str(epoca) + " " + str(errorCuadratico))
                convergio = True
                return np.array(arrayErrorCuadratico), np.array(arrayEpocas), self.pesos1, self.pesosSalida

        if convergio == False:
            exp_scores = np.exp(sensibilidad3)
            probs = exp_scores / np.sum(exp_scores, keepdims=True)
            print("No convergio " + str(errorCuadratico))
            return np.array(arrayErrorCuadratico), np.array(arrayEpocas), self.pesos1, self.pesosSalida

    #    def productoPunto(self, pesos, entradas):
    #        return pesos[0] + pesos[1] * entradas[0] + pesos[2] * entradas[1]

    def funcionSigmoide(self, valor):
        return 1 / (1 + np.exp(-valor))

    def derivadaFuncionSigmoide(self, valor):
        return self.funcionSigmoide(valor) * (1 - self.funcionSigmoide(valor))

    def inicializarPesos(self):
        for i in range(self.neuronasCapa1):
            for j in range(3):
                seed(str(datetime.now()))
                value = random()
                value = -5 + (value * (5 + 5))
                self.pesos1[i][j] = value

        for i in range(self.numeroDeClases):
            for j in range(3):
                seed(str(datetime.now()))
                value = random()
                value = -5 + (value * (5 + 5))
                self.pesosSalida[i][j] = value


"""
if __name__ == "__main__":
    pesos = np.random.rand(3)
    entradas = np.random.randint(5,size=(9,2))
    salidas = np.array([[0,1,0],[0,0,1],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[1,0,0],[1,0,0],[0,0,1],])

    bp = Backpropagation(pesos,entradas,salidas, 5000, 0.09, 0.50, 1, 2, 3)
    errorCuadratico, epocas = bp.backpropagation()

    fig, ax = plt.subplots()
    #Colocamos una etiqueta en el eje Y
    ax.set_ylabel('Error')
    #Colocamos una etiqueta en el eje X
    ax.set_title('Epoca')
    #Creamos la grafica de barras utilizando 'paises' como eje X y 'ventas' como eje y.
    plt.bar(epocas, errorCuadratico)
    plt.show()
    """