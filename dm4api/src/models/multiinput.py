from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate
from tensorflow import split

class MultiInput:
    def __init__(self, config):
        self.config = config
        self.nb_obs = config['nb_obs']
        self.nb_items = config['nb_items']
        self.wlen = config["wlen"]
        if config["nb_slots"]:
            self.nb_slots = config['nb_slots']
        else:
            self.nb_slots = config['nb_actions']
        self.nb_actions = config['nb_actions']

    def f(self, x):
        return split(x,[self.nb_obs[0]-self.nb_items, self.nb_items],1)

    def create_model(self):
        visible = Input(shape=(self.wlen,) + self.nb_obs, name="visible")
        flattened = Flatten(name="flattened")(visible)

        lambda1 = Lambda(lambda x: self.f(x)[0], name="lambda1")(flattened)
        dense1 = Dense(self.nb_actions*2, activation='relu', name="dense1")(lambda1)

        lambda2 = Lambda(lambda x: self.f(x)[1], name="lambda2")(flattened)
        dense2 = Dense(self.nb_slots*2, activation='relu', name="dense2")(lambda2)

        concat1 = Concatenate(name="concat1")([dense1, dense2])

        hidden1 = Dense(self.nb_actions*2, activation='relu', name="hidden1")(concat1)
        output = Dense(self.nb_actions, activation='softmax', name="output")(hidden1)

        model = Model(inputs=visible, outputs=output)

        return model

if __name__ == "__main__":
    miModel = MultiInput({"nb_obs": 3050, "nb_items": 3000, "nb_slots":10, "nb_actions":250})
    model = miModel.create_model()
    model.summary()

