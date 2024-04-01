import json, os
import pandas as pd
import numpy as np

import keras.backend as K
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Input, Concatenate, Cropping2D
from keras.models import Model as KModel, load_model
from keras.optimizers import Adam
from keras import losses
from utils import DiceLoss
# from utils import CustomDiceLoss
# from utils import CustomSobelMseLoss
# from utils import sobelLoss


class Model:
    def __init__(self):
        # self.location = location
        self.parameters = {"name": Model.get_name(), "create": None, "compile": None, "train": None, "predict": None}
        self.models = []
        self.path = None

    @classmethod
    def get_name(cls):
        return "U-Net-Small-Batch-Multi"

    @classmethod
    def get_description(cls):
        return '''
               Simple U-Net based on https://github.com/keras-team/keras/issues/9367.
               '''

    @classmethod
    def get_extents(cls):
        return {"input": 128, "output": 64}

    def create(self, parameters = None):
        if parameters is None:
            self.parameters["create"] = {"num_models" : 10, "num_base_filters" : 8, "num_classes" : 4}
        else:
            self.parameters["create"] = parameters

        for i in range(self.parameters["create"]["num_models"]):
            img_rows = 128
            img_cols = 128
            padding = 32
            num_filters = self.parameters["create"]["num_base_filters"]

            # source: https://github.com/keras-team/keras/issues/9367
            input_shape = (img_rows, img_cols, 4)

            # input holding a single image
            inputs = Input(shape=input_shape, name="input")

            # 128
            c1 = Conv2D(num_filters, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
            c1 = Dropout(0.1)(c1)
            c1 = Conv2D(num_filters, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
            p1 = MaxPooling2D((2, 2))(c1)

            # 64
            c2 = Conv2D(num_filters * 2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
            c2 = Dropout(0.1)(c2)
            c2 = Conv2D(num_filters * 2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
            p2 = MaxPooling2D((2, 2))(c2)

            # 32
            c3 = Conv2D(num_filters * 4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
            c3 = Dropout(0.2)(c3)
            c3 = Conv2D(num_filters * 4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
            p3 = MaxPooling2D((2, 2))(c3)

            # 16
            c4 = Conv2D(num_filters * 8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
            c4 = Dropout(0.2)(c4)
            c4 = Conv2D(num_filters * 8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
            p4 = MaxPooling2D(pool_size=(2, 2))(c4)

            # 8
            c5 = Conv2D(num_filters * 16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
            c5 = Dropout(0.3)(c5)
            c5 = Conv2D(num_filters * 16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

            # 16
            u6 = Conv2DTranspose(num_filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)
            u6 = Concatenate()([u6, c4])
            c6 = Conv2D(num_filters * 8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
            c6 = Dropout(0.2)(c6)
            c6 = Conv2D(num_filters * 8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

            # 32
            u7 = Conv2DTranspose(num_filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
            u7 = Concatenate()([u7, c3])
            c7 = Conv2D(num_filters * 4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
            c7 = Dropout(0.2)(c7)
            c7 = Conv2D(num_filters * 4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

            # 64
            u8 = Conv2DTranspose(num_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
            u8 = Concatenate()([u8, c2])
            c8 = Conv2D(num_filters * 2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
            c8 = Dropout(0.1)(c8)
            c8 = Conv2D(num_filters * 2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

            # 128
            u9 = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(c8)
            u9 = Concatenate()([u9, c1])
            c9 = Conv2D(num_filters, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
            c9 = Dropout(0.1)(c9)
            c9 = Conv2D(num_filters, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

            finalConv = Conv2D(self.parameters["create"]["num_classes"], (1, 1), activation='sigmoid')(c9)
            outputs = Cropping2D(cropping=(padding, padding), data_format=None, name="cropped2")(finalConv)

            model = KModel(inputs=inputs, outputs=outputs)

            self.models.append(model)

    
    def compile(self):
        for model in self.models:
            optimizer = Adam(lr=0.001)
            loss_func = DiceLoss
            # loss_func = losses.binary_crossentropy
            model.compile(optimizer=optimizer, loss=loss_func, metrics=["binary_accuracy"])



    def train(self, training_generators, validation_generators, num_train_images, num_validation_images, batch_size,
              epochs, callbacks_list, location):

        with open(location.__str__() + "/parameters.json", 'w') as outfile:
            json.dump(self.parameters, outfile, indent=4)
            
        for i in range(self.parameters["create"]["num_models"]):
            model = self.models[i]
            training_generator = training_generators[i]
            validation_generator = validation_generators[i]
            callbacks = callbacks_list[i]
            
            # temp = location.__str__()
            # print(temp + "\model_" + str(i))
            # model.save(temp + "\model_" + str(i))

            model.fit_generator(training_generator,
                                 validation_data=validation_generator,
                                 validation_steps=num_validation_images // batch_size,
                                 steps_per_epoch=num_train_images // batch_size,
                                 epochs=epochs,
                                 callbacks=callbacks)

            # TODO: save models when trained and delete them from memory
            # model.save(temp + "model_" + str(i) + ".hdf")
            # print(temp + "\model_" + str(i))
            model.save(location.__str__() + "/model_" + str(i) + ".hdf")
            # model.save(self.location / "model/model_" + str(i) + ".hdf")
   
        
    def predict(self, X, batch_size):
        # if self.parameters is None:
        #     self.parameters["create"] = {"num_models" : 10, "num_base_filters" : 8, "num_classes" : 4}
        
        # Y_pred_accum = np.zeros((batch_size, 64, 64, 1), np.float32)
        Y_pred_accum = np.zeros((batch_size, 64, 64, self.parameters["create"]["num_classes"]), np.float32)


        for m in range(self.parameters["create"]["num_models"]):
            model = self.models[m]

            Y_pred = model.predict(X, batch_size=batch_size)
            Y_pred_accum += Y_pred

        img_result = Y_pred_accum / self.parameters["create"]["num_models"]

        return img_result

    def save(self, path):
        with open(str(path) + "/parameters.json", 'w') as outfile:
            json.dump(self.parameters, outfile, indent=4)

        for i in range(self.parameters["create"]["num_models"]):
            self.models[i].save(str(path) + "/model_" + str(i) + ".hdf")


    def load(self, path):
        self.models = []
        with open(str(path) + "/parameters.json") as metadata_json:
            self.parameters = json.load(metadata_json)
        print(self.parameters)

        for i in range(self.parameters["create"]["num_models"]):
            # self.models.append(load_model(str(path) + "/model_" + str(i) + ".hdf", custom_objects={'sobelLoss': sobelLoss}))
            self.models.append(load_model(str(path) + "/model_" + str(i) + ".hdf", custom_objects={'DiceLoss': DiceLoss}))
            # self.models.append(load_model(str(path) + "/model_" + str(i) + ".hdf", custom_objects={'CustomSobelMseLoss': CustomSobelMseLoss}))
            # self.models.append(load_model(str(path) + "/model_" + str(i) + ".hdf"))

    def get_log(self):
        log_path = self.path + "/training.log"
        return pd.read_csv(log_path)

