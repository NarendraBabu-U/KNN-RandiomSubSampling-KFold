from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('iris.csv',usecols=[1,3],delimiter=',')
cat = np.loadtxt('iris.csv',usecols=[4],dtype=str,delimiter=',')
cn= []
for i in cat:
         if i=="Iris-setosa":
                  cn.append(0)
         elif i=="Iris-versicolor":
                  cn.append(1)
         else:
                  cn.append(2)

colors  =  ListedColormap(['r', 'g', 'b'])
Xcolors = ListedColormap(['k', 'y', 'm'])
xstart, xend = data[:, 0].min() - 1, data[:, 0].max() + 1
ystart, yend = data[:, 1].min() - 1, data[:, 1].max() + 1
xgrid, ygrid = np.meshgrid(np.arange(xstart, xend, 0.1), np.arange(ystart, yend, 0.1))
xy_grid = np.c_[xgrid.ravel(), ygrid.ravel()]
pred = []
for i in xy_grid:
         distance = []
         for j in data:
                  distance.append(np.linalg.norm(j-i))
         pred.append(cn[np.argmin(distance)])
pred = np.asarray(pred)
pred = pred.reshape(xgrid.shape)
plt.figure()
plt.pcolormesh(xgrid, ygrid, pred, cmap=colors )
plt.scatter(data[:, 0], data[:, 1], c=cn, cmap=Xcolors)
plt.axis('tight')    
plt.xlabel('Sepal Length')
plt.ylabel('Petal Width')
plt.show()
