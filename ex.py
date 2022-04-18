from MyModel2 import MyModel2
import tensorflow.keras as keras
from BilinearModel import BilinearModel


inputs = keras.Input(shape=(224, 224, 3), name="my_input")
mymodel = MyModel2(inputs).CreateMyModel()
mymodel.summary()

inputs = keras.Input(shape=(224, 224, 3), name="my_input")
mymodel = BilinearModel(inputs).CreateMyModel()
mymodel.summary()

