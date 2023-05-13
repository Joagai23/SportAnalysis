from keras import metrics
import numpy as np

cat_acc = metrics.CategoricalAccuracy()
yTrue = np.array([[1,0,0,0],[1,0,0,0]]) 
yPred = np.array([[6.8764354e-04, 2.6750509e-04, 2.4782584e-04, 9.9879706e-01], [9.7892797e-01, 2.6669452e-04, 2.1525579e-05, 2.0783830e-02]])
cat_acc.update_state(yTrue, yPred)
print(cat_acc.result().numpy())

cat_acc.rese