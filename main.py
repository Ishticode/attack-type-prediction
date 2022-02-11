import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

full_table = pd.read_csv("globalterrorismdb_0718dist.csv", encoding="latin1")
specific_table = full_table[["summary", "attacktype1_txt"]]
specific_table.dropna(inplace=True)
x = specific_table["summary"]
y = specific_table["attacktype1_txt"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
max_vocab = 10000
max_length = 40
textvex = keras.layers.TextVectorization(max_tokens=max_vocab,
                                         output_sequence_length = max_length)

textvex.adapt(x_train)

embedding = keras.layers.Embedding(input_dim=max_vocab,
                             output_dim=128,
                             embeddings_initializer="uniform",
                             input_length=max_length)

inputs = keras.layers.Input(shape=(1,), dtype="string") # inputs are 1-dimensional strings

x = textvex(inputs) # turn the input text into numbers
#x = tf.keras.layers.Lambda(uEmbeddingLayer, output_shape=None, mask=None, arguments=None)(x)
 # create an embedding of the numerized numbers
x=embedding(x)

x = keras.layers.Conv1D(filters=4,kernel_size=3,padding="valid")(x)
x = keras.layers.GlobalAveragePooling1D()(x) # lower the dimensionality of the embedding (try running the model without this layer and see what happens)
outputs = keras.layers.Dense(9, activation="sigmoid")(x) # create the output layer, want binary outputs so use sigmoid activation
model_1 = tf.keras.Model(inputs, outputs, name="model_1_dense")

model_1.compile(loss= keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
