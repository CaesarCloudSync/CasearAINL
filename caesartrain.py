# -*- coding: utf-8 -*-
import os
import json
#import shutil
import pickle
import warnings
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

class CaesarNLTrain:
    def train(traindf,validdf,testdf,examples,history_filename = "history.png"):
        trainfeatures=traindf.copy()
        trainlabels=trainfeatures.pop("intent")

        trainfeatures=trainfeatures.values



        """One-Hot-Encoding of class-labels:"""



        binarizer=LabelBinarizer()
        trainlabels=binarizer.fit_transform(trainlabels.values)


        """Preprocess test- and validation data in the same way as it has been done for training-data:"""

        testfeatures=testdf.copy()
        testlabels=testfeatures.pop("intent")
        validfeatures=validdf.copy()
        validlabels=validfeatures.pop("intent")

        testfeatures=testfeatures.values
        validfeatures=validfeatures.values

        testlabels=binarizer.transform(testlabels.values)
        validlabels=binarizer.transform(validlabels.values)
        pickle.dump(binarizer, open('caesarmodel/labelbinarizer.pkl', 'wb'))

        bert_model_name = 'small_bert/bert_en_uncased_L-8_H-512_A-8' 
        with open("caesarberthubmodels/bert_to_handle.json") as f:
            map_name_to_handle = json.load(f)
        with open("caesarberthubmodels/bert_to_preprocess.json") as f:
            map_model_to_preprocess =  json.load(f)



        tfhub_handle_encoder = map_name_to_handle[bert_model_name]
        tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

        print(f'BERT model selected           : {tfhub_handle_encoder}')
        print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')



        bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)


        trainfeatures[0]

        text_test = trainfeatures[0]
        text_preprocessed = bert_preprocess_model(text_test)

        bert_model = hub.KerasLayer(tfhub_handle_encoder)

        bert_results = bert_model(text_preprocessed)



        def build_classifier_model():
            text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
            preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
            encoder_inputs = preprocessing_layer(text_input)
            encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
            outputs = encoder(encoder_inputs)
            net = outputs['pooled_output']
            net = tf.keras.layers.Dropout(0.1)(net)
            net = tf.keras.layers.Dense(7, activation=None, name='classifier')(net)
            return tf.keras.Model(text_input, net)

        """Let's check that the model runs with the output of the preprocessing model."""

        classifier_model = build_classifier_model()
        bert_raw_result = classifier_model(tf.constant(trainfeatures[0]))
        print(tf.keras.activations.softmax(bert_raw_result))

        """The output is meaningless, of course, because the model has not been trained yet.

        Let's take a look at the model's structure.
        """

        classifier_model.summary()

        """## Model training

        You now have all the pieces to train a model, including the preprocessing module, BERT encoder, data, and classifier.

        Since this is a non-binary classification problem and the model outputs probabilities, you'll use `losses.CategoricalCrossentropy` loss function.
        """

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics = tf.metrics.CategoricalAccuracy()

        """### Loading the BERT model and training

        Using the `classifier_model` you created earlier, you can compile the model with the loss, metric and optimizer.
        """

        epochs=5
        optimizer=tf.keras.optimizers.Adam(1e-5)
        classifier_model.compile(optimizer=optimizer,
                                loss=loss,
                                metrics=metrics)

        """Note: training time will vary depending on the complexity of the BERT model you have selected."""

        print(f'Training model with {tfhub_handle_encoder}')
        history = classifier_model.fit(x=trainfeatures,y=trainlabels,
                                    validation_data=(validfeatures,validlabels),
                                    batch_size=32,
                                    epochs=epochs)
        classifier_model.save("caesarmodel/caesarnl.h5")

        """### Evaluate the model

        Let's see how the model performs. Two values will be returned. Loss (a number which represents the error, lower values are better), and accuracy.
        """

        loss, accuracy = classifier_model.evaluate(testfeatures,testlabels)

        print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy}')

        """### Plot the accuracy and loss over time

        Based on the `History` object returned by `model.fit()`. You can plot the training and validation loss for comparison, as well as the training and validation accuracy:
        """

        history_dict = history.history
        print(history_dict.keys())

        acc = history_dict['categorical_accuracy']
        val_acc = history_dict['val_categorical_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)
        fig = plt.figure(figsize=(10, 8))
        fig.tight_layout()

        plt.subplot(2, 1, 1)
        # "bo" is for "blue dot"
        plt.plot(epochs, loss, 'r', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.grid(True)
        # plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        plt.legend(loc='lower right')
        plt.savefig(f"caesartrainperformance/{history_filename}")

        """In this plot, the red lines represents the training loss and accuracy, and the blue lines are the validation loss and accuracy.

        Classifying arbitrary instructions:
        """

        def print_my_examples(inputs, results):
            result_for_printing = \
                [f'input: {inputs[i]:<30} : estimated intent: {results[i]}'
                                    for i in range(len(inputs))]
            print(*result_for_printing, sep='\n')
            print()




        results = tf.nn.softmax(classifier_model(tf.constant(examples)))

        binarizer.classes_

        intents=binarizer.inverse_transform(results.numpy())

        print_my_examples(examples, intents)
if __name__ == "__main__":
    examples = [
            'play a song from U2',  # this is the same sentence tried earlier
            'Will it rain tomorrow',
            'I like to hear greatist hits from beastie boys',
            'I like to book a table for 3 persons',
            '5 stars for machines like me'
        ]
    datafolder="intentdata/"
    trainfile=datafolder+"train.csv"
    testfile=datafolder+"test.csv"
    validfile=datafolder+"valid.csv"

    """Next, the downloaded .csv-files for training, validation and test are imported into pandas dataframes:"""

    traindf = pd.read_csv(trainfile)
    validdf = pd.read_csv(validfile)
    testdf = pd.read_csv(testfile)
    
    CaesarNLTrain.train(traindf,validdf,testdf,examples,history_filename = "")

