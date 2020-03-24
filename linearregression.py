import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt

data_x = [338.,333.,328.,207.,226.,25.,179.,60.,208.,606.]
data_y = [640.,633.,619.,393.,428.,27.,193.,66.,226.,1591]

#data_x = [38.,33.,32.,20.,22.,25.,17.,60.,20.,65.]
#data_y = [76.,66.,64.,40.,44.,50.,34.,120.,40.,130.]
#data_y = b + w1 * data_x + w2 * data_x ** 2


x = np.arange(-200,100,1)
y = np.arange(-15,15,0.1)
z = np.zeros((len(x),len(y)))
X,Y = np.meshgrid(x,y)


for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        z[j][i] = 0
        for n in range(len(data_x)):
            z[j][i] = z[j][i] + (data_y[n] - b - w * data_x[n])**2
        z[j][i] = z[j][i]/len(data_x)


	
b = 1
w1 = 1
w2 = 1
lr = 1
iteration = 100000

b_history = [b]
w1_history = [w1]
w2_history = [w2]

lr_b = 0
lr_w1 = 0
lr_w2 = 0

for i in range(iteration):
    b_grad = 0.0
    w1_grad = 0.0
    w2_grad = 0.0
    for n in range(len(data_x)):
        b_grad = b_grad - 2.0 * (data_y[n] - b - w1 * data_x[n] - w2 * data_x[n] * data_x[n])* 1.0
        w1_grad = w1_grad - 2.0 * (data_y[n] - b - w1 * data_x[n]- w2 * data_x[n] * data_x[n])* data_x[n]
        w2_grad = w2_grad - 2.0 * (data_y[n] - b - w1 * data_x[n]- w2 * data_x[n] * data_x[n])* data_x[n] * data_x[n]
        #print b_grad
    lr_b = lr_b + b_grad ** 2
    lr_w1 = lr_w1 + w1_grad ** 2
    lr_w2 = lr_w2 + w2_grad **2
    b = b - lr/np.sqrt(lr_b) * b_grad
    w1 = w1 - lr/np.sqrt(lr_w1) * w1_grad
    w2 = w2 - lr/np.sqrt(lr_w2) * w2_grad	
    b_history.append(b)
    w1_history.append(w1)
    w2_history.append(w2)
    #print i,b,w1,w2
print b
print w1
print w2	
#err = 0.0
#for exam in range(len(data_x)):
    #err = err + (y[exam]-(b + w * x[exam])) ** 2
    #print err
	
plt.contourf(x,y,z,50,alpha = 0.5,cmp = plt.get_cmap('jet'))
plt.plot([-188.4],[2.67],'x',ms = 12, markeredgewidth =3,color ='red')
plt.plot(b_history,w1_history,'o-',ms = 3,lw =1.5,color = 'black')
#plt.xlim(-200,-100)
plt.xlim(-200,100)
plt.ylim(-5,5)
plt.xlabel(r'$b$',fontsize=16)
plt.ylabel(r'$w$',fontsize=16)

plt.show()







