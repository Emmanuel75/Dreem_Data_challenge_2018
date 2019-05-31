
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import to_categorical

df = pd.read_excel('C:\\Users\\i053131\\Desktop\\Epilepsie\\Dreem\\data\\interim\\featuresTrain.xlsx')


training, test  = train_test_split(df, test_size=0.2, random_state=42)
X = training.iloc[:,:-1]
X_train = X.values
y = training.iloc[:,-1]
y_train = to_categorical(y.values, num_classes=5)
X_test = test.iloc[:,:-1].values
y_true = test.iloc[:,-1].values
y_test = to_categorical(y_true, num_classes=5)

sns.countplot(x=y, data=training)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=63))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=128)
score = model.evaluate(X_test, y_test, batch_size=128)
y_pred = pd.DataFrame(model.predict_classes(X_test, batch_size=128))

sns.countplot(x=0, data=y_pred)