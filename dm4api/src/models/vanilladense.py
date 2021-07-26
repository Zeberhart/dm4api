from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten


class VanillaDense:
    def __init__(self, config):
        self.config = config
        self.nb_obs = config['nb_obs']
        self.nb_actions = config['nb_actions']
        self.wlen = config["wlen"]
        print(self.nb_actions)

    def create_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(self.wlen,) + self.nb_obs))
        model.add(Dense(100))
        model.add(Activation('sigmoid'))
        model.add(Dense(self.nb_actions))
        model.add(Activation('softmax'))
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

