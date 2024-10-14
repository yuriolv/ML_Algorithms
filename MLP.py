import numpy as np
import pandas as pd

#criação da própria função de codificação oneHot
def one_hot(Y):
    #matriz de 0's com linhas = qtd_exemplos e colunas = n_classes
    one_hot = np.zeros((Y.size, Y.max() + 1))
    #em cada linha i, substituir one_hot[Y] = Y[i]
    one_hot[np.arange(Y.size), Y] = 1
    return one_hot.T

class MLP:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.w1 = np.random.rand(10,784) - 0.5 #weights: input -> hidden1
        self.b1 = np.random.rand(10, 1) - 0.5 #bias: hidden1

        self.w2 = np.random.rand(10, 10) - 0.5 #weights: hidden1 -> hidden2
        self.b2 = np.random.rand(10, 1) - 0.5 #bias:hidden2

        self.w3 = np.random.rand(10, 10) - 0.5 #weights: h2 -> h3
        self.b3 = np.random.rand(10, 1) - 0.5 #bias: h3

        self.learning_rate = learning_rate 
        self.epochs = epochs
        
    def ReLU(self, Z):
        return np.maximum(Z, 0) #máximo entre Z e 0
    
    def ReLU_deriv(self, Z):
        return Z > 0
    
    def softmax(self, Z):
        A = np.exp(Z)/ sum(np.exp(Z)) #exponencial de cada elemento de saída / soma 
        return A 
    
    def foward_prop(self, X):
        #iremos passar a entrada X e multiplicar pelos pesos + bias e colocar na ativação
        Z1 = self.w1.dot(X) + self.b1
        A1 = self.ReLU(Z1)
        #mesma coisa para a segunda camada, porém com a saída de A1
        Z2 = self.w2.dot(A1) + self.b2
        A2 = self.ReLU(Z2)

        Z3 = self.w3.dot(A2) + self.b3
        A3 = self.softmax(Z3)
        return Z1, A1, Z2, A2, A3
        
    """ em poucas palavras, se faz a propagação em sentido contrário, descobre-se o 
    quanto os pessos erraram e ajusta-se com base nisso. """
    def backward_prop(self,Z1,Z2,A1,A2,A3,X, Y):
        one_hot_Y = one_hot(Y)
        loss3 = A3 - one_hot_Y #erro da 2 camada oculta
        dw3 = 1 / Y.size * loss3.dot(A2.T) #derivada da segunda camada para ajustar 
        db3 = 1 / Y.size * np.sum(loss3) #ajuste no bias da segunda camada (média)

        #erro da segunda camada multiplicado pelos pesos e multiplicamos à derivada da primeira camada
        loss2 = self.w3.T.dot(loss3) * self.ReLU_deriv(Z2) #influência da primeira camada na saída (back)
        dw2 = 1 / Y.size * loss2.dot(A1.T) #ajuste dos pesos
        db2 = 1 / Y.size * np.sum(loss2) #ajuste do bias primeira camada
        
        loss1 = self.w2.T.dot(loss2) * self.ReLU_deriv(Z1) #influência da primeira camada na saída (back)
        dw1 = 1 / Y.size * loss1.dot(X.T) #ajuste dos pesos
        db1 = 1 / Y.size * np.sum(loss1) #ajuste do bias primeira camada
        
        return dw1, db1, dw2, db2, dw3, db3

    def update_params(self, dw1, db1, dw2, db2, dw3, db3):
        self.w1 = self.w1 - self.learning_rate * dw1
        self.b1 = self.b1 - self.learning_rate * db1

        self.w2 = self.w2 - self.learning_rate * dw2
        self.b2 = self.b2 - self.learning_rate * db2

        self.w3 = self.w3 - self.learning_rate * dw3
        self.b3 = self.b3 - self.learning_rate * db3
    
    def get_predict(self, A3):
        return np.argmax(A3, 0)
    
    def get_accuracy(self, predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size
    
    def fit(self, X, Y):
        for i in range(self.epochs):
            for row, target in zip(X,Y):
                Z1, A1,Z2, A2, A3 = self.foward_prop(row)
                dw1, db1, dw2, db2, dw3, db3 = self.backward_prop(Z1,Z2,A1,A2,A3,row, target)
                self.update_params(dw1, db1, dw2, db2,dw3, db3)

            if i % 10 == 0:
                print("Epoch: ", i)
                predictions = self.get_predict(A3)
                print(self.get_accuracy(predictions, Y))

