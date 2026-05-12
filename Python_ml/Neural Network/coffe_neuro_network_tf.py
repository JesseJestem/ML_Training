import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

#plot building
def plt_roast(X, Y, X_test=None, yhat=None):

    good = Y[:, 0] == 1
    bad = Y[:, 0] == 0

    # Bad Roast: blue circles
    plt.scatter(
        X[bad, 0],
        X[bad, 1],
        marker='o',
        facecolors='none',
        edgecolors='blue',
        label='Bad Roast'
    )

    # Good Roast: red crosses
    plt.scatter(
        X[good, 0],
        X[good, 1],
        marker='x',
        c='red',
        linewidths=2,
        label='Good Roast'
    )

    # Predictions: stars
    if X_test is not None and yhat is not None:
        good_pred = yhat[:, 0] == 1
        bad_pred = yhat[:, 0] == 0

        # Predicted Good Roast: red stars
        plt.scatter(
            X_test[good_pred, 0],
            X_test[good_pred, 1],
            marker='*',
            s=250,
            c='red',
            edgecolors='black',
            label='Predicted Good Roast'
        )

        # Predicted Bad Roast: blue stars
        plt.scatter(
            X_test[bad_pred, 0],
            X_test[bad_pred, 1],
            marker='*',
            s=250,
            c='blue',
            edgecolors='black',
            label='Predicted Bad Roast'
        )

    plt.title("Coffee Roasting")
    plt.xlabel("Temperature\n(Celsius)")
    plt.ylabel("Duration\n(minutes)")

    # Guide lines
    plt.axvline(x=175, color='purple', linewidth=0.8)
    plt.axhline(y=12, color='purple', linewidth=0.8)
    plt.plot([175, 260], [14.7, 12.0], color='purple', linewidth=0.8)

    plt.xlim(150, 285)
    plt.ylim(11.4, 17.3)

    plt.legend(loc="upper right")
    plt.show()

#data set
X = np.array([
    [185.32, 12.69],
    [259.92, 11.87],
    [231.01, 14.41],
    [175.37, 11.72],
    [187.12, 14.13],
    [225.91, 12.1],
    [208.41, 14.18],
    [207.08, 14.03],
    [280.6, 14.23],
    [202.87, 12.25],
    [196.7, 13.54],
    [270.31, 14.6],
    [192.95, 15.2],
    [213.57, 14.28],
    [164.47, 11.92],
    [177.26, 15.04],
    [241.77, 14.9],
    [237.0, 13.13],
    [219.74, 13.87],
    [266.39, 13.25],
    [270.45, 13.95],
    [261.96, 13.49],
    [243.49, 12.86],
    [220.58, 12.36],
    [163.59, 11.65],
    [244.76, 13.33],
    [271.19, 14.84],
    [201.99, 15.39],
    [229.93, 14.56],
    [204.97, 12.28],
    [173.19, 12.22],
    [231.51, 11.95],
    [152.69, 14.83],
    [163.42, 13.3],
    [215.95, 13.98],
    [218.04, 15.25],
    [251.3, 13.8],
    [233.33, 13.53],
    [280.24, 12.41],
    [243.02, 13.72],
    [155.67, 12.68],
    [275.17, 14.64],
    [151.73, 12.69],
    [151.32, 14.81],
    [164.9, 11.73],
    [282.55, 13.28],
    [192.98, 11.7],
    [202.6, 12.96],
    [220.67, 11.53],
    [169.97, 12.34],
    [209.47, 12.71],
    [232.8, 12.64],
    [272.8, 15.35],
    [158.02, 12.34],
    [226.01, 14.58],
    [158.64, 12.24],
    [211.66, 14.17],
    [271.95, 14.97],
    [257.16, 11.71],
    [281.85, 13.96],
    [161.63, 12.52],
    [233.8, 13.04],
    [210.29, 14.72],
    [261.24, 13.69],
    [256.98, 13.12],
    [281.56, 13.92],
    [280.64, 11.68],
    [269.16, 13.74],
    [246.34, 12.27],
    [224.07, 12.66],
    [164.24, 11.51],
    [272.42, 14.18],
    [177.68, 12.53],
    [212.86, 14.77],
    [165.88, 15.37],
    [277.43, 12.48],
    [236.51, 12.94],
    [244.14, 11.85],
    [213.45, 13.85],
    [234.57, 14.27],
    [270.34, 12.47],
    [170.68, 13.06],
    [226.79, 15.34],
    [245.92, 14.45],
    [281.32, 12.57],
    [185.03, 13.19],
    [189.88, 14.1],
    [278.48, 12.11],
    [219.92, 14.21],
    [216.58, 15.15],
    [249.48, 15.03],
    [165.09, 12.28],
    [158.87, 14.82],
    [279.98, 11.56],
    [256.55, 14.41],
    [272.61, 12.58],
    [246.49, 12.45],
    [160.26, 14.48],
    [155.7, 14.3],
    [188.27, 13.45],
    [270.36, 12.47],
    [213.22, 12.92],
    [175.7, 13.39],
    [174.52, 14.7],
    [233.0, 12.63],
    [281.37, 12.88],
    [240.62, 14.43],
    [185.81, 11.55],
    [270.5, 15.33],
    [172.98, 12.11],
    [208.41, 13.89],
    [283.51, 15.35],
    [283.36, 12.48],
    [230.85, 13.24],
    [181.24, 11.76],
    [172.78, 12.93],
    [161.88, 12.1],
    [156.03, 13.99],
    [216.52, 12.47],
    [221.06, 13.2],
    [238.99, 15.23],
    [197.69, 14.08],
    [179.55, 15.26],
    [233.39, 12.13],
    [184.7, 12.14],
    [174.18, 12.73],
    [261.11, 13.33],
    [187.42, 13.18],
    [186.1, 14.43],
    [157.94, 12.66],
    [193.64, 12.23],
    [249.65, 12.22],
    [190.56, 11.73],
    [252.0, 12.96],
    [238.55, 12.37],
    [152.94, 12.79],
    [255.17, 14.85],
    [197.09, 14.89],
    [156.8, 13.59],
    [184.75, 13.26],
    [179.92, 15.07],
    [190.79, 15.28],
    [164.73, 13.22],
    [209.87, 14.34],
    [196.58, 13.47],
    [159.51, 12.74],
    [247.87, 11.92],
    [212.44, 12.45],
    [172.34, 11.99],
    [259.87, 14.25],
    [201.23, 13.07],
    [248.34, 13.92],
    [273.66, 15.18],
    [215.09, 14.14],
    [223.53, 12.74],
    [211.22, 14.38],
    [224.61, 14.03],
    [215.75, 15.31],
    [254.82, 12.02],
    [259.9, 15.17],
    [260.25, 12.87],
    [199.67, 12.47],
    [157.52, 13.39],
    [264.81, 14.58],
    [239.4, 14.89],
    [238.98, 12.39],
    [258.43, 12.97],
    [270.16, 12.81],
    [162.41, 14.42],
    [164.53, 14.98],
    [205.61, 14.62],
    [157.1, 13.68],
    [241.38, 12.02],
    [232.13, 12.07],
    [191.04, 12.96],
    [233.64, 12.02],
    [174.95, 14.63],
    [246.64, 13.32],
    [188.07, 14.27],
    [213.16, 12.75],
    [268.08, 12.31],
    [258.58, 13.97],
    [237.21, 14.23],
    [251.02, 15.02],
    [274.28, 12.52],
    [172.12, 15.09],
    [177.52, 12.39],
    [258.71, 15.36],
    [264.01, 13.57],
    [200.71, 15.45],
    [249.37, 14.02],
    [151.5, 12.28],
    [151.82, 15.13],
    [181.92, 12.18],
    [228.65, 12.31],
    [223.78, 15.3],
    [266.63, 12.48],
    [273.68, 13.1],
    [220.61, 12.8],
    [284.99, 12.73],
])

Y = np.array([
    [1.],
    [0.],
    [0.],
    [0.],
    [1.],
    [1.],
    [0.],
    [0.],
    [0.],
    [1.],
    [1.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [1.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [1.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [1.],
    [0.],
    [0.],
    [1.],
    [1.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [1.],
    [1.],
    [0.],
    [0.],
    [1.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [1.],
    [1.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [1.],
    [0.],
    [1.],
    [1.],
    [0.],
    [1.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [1.],
    [1.],
    [0.],
    [0.],
    [0.],
    [1.],
    [1.],
    [0.],
    [0.],
    [1.],
    [0.],
    [0.],
    [1.],
    [0.],
    [0.],
    [0.],
    [1.],
    [0.],
    [0.],
    [0.],
    [0.],
    [1.],
    [0.],
    [0.],
    [0.],
    [0.],
    [1.],
    [0.],
    [0.],
    [1.],
    [0.],
    [0.],
    [1.],
    [0.],
    [0.],
    [0.],
    [1.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [1.],
    [0.],
    [0.],
    [0.],
    [1.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [1.],
    [1.],
    [1.],
    [1.],
    [0.],
    [0.],
    [1.],
    [1.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [1.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [0.],
    [1.],
    [1.],
    [0.],
    [0.],
    [0.],
    [1.],
    [0.],
])

#checking of data set
print(X.shape, Y.shape)
print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

#show plot
plt_roast(X,Y)

#copy our data to increase the training set size and reduce the number of training epochs
Xt = np.tile(Xn,(1000,1))
Yt= np.tile(Y,(1000,1))
print(Xt.shape, Yt.shape)

tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(2,)), #2 layers
        Dense(3, activation='sigmoid', name = 'layer1'),
        Dense(1, activation='sigmoid', name = 'layer2')
     ]
)

model.summary() #provides a description of the network

#show numbers parameters of network
L1_num_params = 2 * 3 + 3   # W1 parameters  + b1 parameters
L2_num_params = 3 * 1 + 1   # W2 parameters  + b2 parameters
print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params  )

#show setup
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)


#start our model
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(
    Xt,Yt,
    epochs=10,
)

#lets see new set
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

#setup new set
W1 = np.array([
    [-8.94,  0.29, 12.89],
    [-0.17, -7.34, 10.79]] )
b1 = np.array([-9.87, -9.28,  1.01])
W2 = np.array([
    [-31.38],
    [-27.86],
    [-32.79]])
b2 = np.array([15.54])

# Replace the weights from your trained model with
# the values above.
model.get_layer("layer1").set_weights([W1,b1])
model.get_layer("layer2").set_weights([W2,b2])

#show raw predictions
X_test = np.array([
    [200,13.9],  # positive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)

#show result
yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")

#final prediction plots
X_test = np.array([
    [200, 13.9],  # expected Good Roast
    [200, 17.0]   # expected Bad Roast
])

#try to predict
X_test = np.array([
    [200, 13.9],
    [200, 17.0]
])

X_testn = norm_l(X_test)

predictions = model.predict(X_testn)
yhat = (predictions >= 0.5).astype(int)

plt_roast(X, Y, X_test, yhat)