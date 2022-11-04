import spacy
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.preprocessing import LabelBinarizer

def print_my_examples(inputs, results):
  result_for_printing = \
    [f'input: {inputs[i]:<30} : estimated intent: {results[i]}'
                         for i in range(len(inputs))]
  print(*result_for_printing, sep='\n')
  print()


examples = [
    'play a song from U2',  # this is the same sentence tried earlier
    'Will it rain tomorrow',
    'I like to hear greatist hits from beastie boys',
    'I like to book a table for 3 persons',
    '5 stars for machines like me',
    'play a boogie wit da hoodie',
    "play Bob's favorite song"
]
#nlp = spacy.load("en_core_web_sm")
classifier_model = tf.keras.models.load_model('caesarmodel/caesarnl.h5',custom_objects={'KerasLayer':hub.KerasLayer})

# Show the model architecture
results = tf.nn.softmax(classifier_model(tf.constant(examples)))
with open("caesarmodel/labelbinarizer.pkl","rb") as f:
    binarizer = pickle.load(f)


intents=binarizer.inverse_transform(results.numpy())
sentence_intents = dict(zip(examples,intents))
print(sentence_intents)
#print_my_examples(examples, intents)

# TODO AIM - Implement Chatbot Gossip to Caesar
# 1. Add data to datasets train | valid | test 
#     a. then clean labels
# 2. Augment data to provide more potential possibilites
# 3. Use BERT to match input with the response
# Command Labels - AddToPlaylist | GetWeather -> API -> user
# Conversation Labes - Greeting | Goodbye  -> BERTNN: input:"hello" => response:"hi there, I am caesar" -> user

# TODO AIM - Single names of songs artists like "play a boogie" and it will play a boogie's music.
# 1. Idea one - NER detect the named entities
# 2. Create new Neural Network that detects that. * Have to determine the relationship between the entites 


