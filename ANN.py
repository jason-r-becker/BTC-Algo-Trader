import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler

rng = np.random.RandomState(12)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

algorithm = 'LR'  # {'ANN': Artificial Nerual Network, 'LR': Logistic Regression}

print(pd.to_datetime('2017 9 15'))

# Load data
df = pd.read_csv('Training_Data/btc final.csv', index_col=0)
df = df[df.index <= df.index[1100]]
n = len(df.index)  # number of samples
train = df.ix[:int(0.85 * n)]  # train with 85% of samples
n_train = len(train.index)  # number of training samples
valid = train.ix[int(0.85 * n_train):]  # validate with 20% of training samples
train = train.ix[:int(0.85 * n_train)]  # train with 80% of training samples
test = df.ix[int(0.85 * n):]  # test on 15% of samples

print(train.tail())
print(test.tail())
quit()

# print('{} Training samples'.format(len(train.index)))
# print('{} Validation samples'.format(len(valid.index)))
# print('{} Testing samples'.format(len(test.index)))


# Get training and testing sets
X_train = train.drop(['return', 'move'], axis=1)
X_valid = valid.drop(['return', 'move'], axis=1)
X_test = test.drop(['return', 'move'], axis=1)
y_train = train['move']
y_valid = valid['move']
y_test = test['move']

# Scale the data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


if algorithm == 'ANN':
    from keras.callbacks import ModelCheckpoint
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, BatchNormalization

    # Build model
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1]),

        BatchNormalization(),
        Dense(64, activation='tanh'),

        Dropout(0.2),
        BatchNormalization(),
        Dense(128, activation='tanh'),

        Dropout(0.3),
        BatchNormalization(),
        Dense(256, activation='tanh'),

        Dropout(0.4),
        BatchNormalization(),
        Dense(256, activation='tanh'),

        Dropout(0.4),
        BatchNormalization(),
        Dense(256, activation='tanh'),

        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])

    # model.summary()


    # Train Model
    def train_ANN():
        # Compile model
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        # Train model
        epochs = 50
        checkpointer = ModelCheckpoint(filepath='Saved_Models/ANN_Weights.hdf5', verbose=0, save_best_only=True)
        model.fit(X_train, y_train, validation_data=(X_valid, y_valid), verbose=0, epochs=epochs, batch_size=20,
                  callbacks=[checkpointer])

    # train_ANN()

    # Load best model
    # model.load_weights('Saved_Models/ANN_Weights.hdf5')
    model.load_weights('Saved_Models/ANN_Weights_57.3.hdf5')

elif algorithm == 'LR':
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    clf = LogisticRegression(random_state=0)

    # Select optimal model using hyper-paramatrization
    parameters = {'solver': ['liblinear', 'newton-cg', 'lbfgs'], 'C': np.logspace(-3, 2, 15)}
    grid_obj = GridSearchCV(clf, param_grid=parameters)
    grid_fit = grid_obj.fit(X_train, y_train)
    model = grid_fit.best_estimator_

else:
    print("Error: Please select valid algorithm.\n Select 'ANN' for neural network or 'LR' for logistic regression.")
    quit()

# Test model
train_prob = np.squeeze(np.asarray((model.predict(X_train))))
train_pred = np.squeeze(np.asarray(np.round(train_prob)))
train_acc = np.mean(train_pred == y_train)

test_prob = np.squeeze(np.asarray((model.predict(X_test))))
test_pred = np.squeeze(np.asarray(np.round(test_prob)))
test_acc = np.mean(test_pred == y_test)

print('Train Accuracy:\t{:.3f}'.format(train_acc * 100))
print('Test Accuracy:\t{:.3f}'.format(test_acc * 100))

# Save results
results = pd.DataFrame({'probs': test_prob, 'preds': test_pred, 'actual': np.asarray(y_test)})
results.to_csv('Model Results.csv')
