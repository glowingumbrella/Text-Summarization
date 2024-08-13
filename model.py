# Parts of our encoder-decoder model were adapted from https://towardsdatascience.com/text-summarization-from-scratch-using-encoder-decoder-network-with-attention-in-keras-5fa80d12710e

import tensorflow as tf
import pickle
import numpy as np
import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping


#utility to load paramaters
def load_parameters(name = ''):
    with open(f'src/parameters{name}.pickle', 'rb') as f:
        return pickle.load(f)
        
#utility to load dataset
def load_ds(name = ''):
    with open(f'src/dataset{name}.pickle', 'rb') as f:
        return pickle.load(f)

# encoder-decoder model class
class TextSummarizerModel():
    def __init__(self, units, output_dim, x_size, y_size, x_tokenizer, y_tokenizer,max_text_len, max_summary_len, embeddings = None):
        self.units = units
        self.output_dim = output_dim
        self.x_size = x_size
        self.y_size = y_size
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len
        self.x_tokenizer = x_tokenizer
        self.y_tokenizer = y_tokenizer
        self.embeddings = embeddings

        #Encoder Setup-----------------------------------------------------------------------------------------------------------------------# 
        self.encoder_inputs = Input(shape=(self.max_text_len,))
        #embedding layer
        self.encoder_embedding_layer =  Embedding(x_size, self.output_dim,trainable=True, name = 'Encoder_Embedding')

        #triple stacked lstms
        self.encoder_lstm1 = LSTM(self.units, return_sequences=True, name = 'Encoder_LSTM1')
        self.encoder_lstm2 = LSTM(self.units, return_sequences=True, name = 'Encoder_LSTM2')
        self.encoder_lstm3 = LSTM(self.units, return_sequences=True, return_state=True, name = 'Encoder_LSTM3')


        # Decoder Setup-----------------------------------------------------------------------------------------------------------------------#
        self.decoder_inputs = Input(shape=(None,))

        #embedding layer
        self.decoder_embedding_layer = Embedding(y_size, self.output_dim,trainable=True, name = 'Decoder_Embedding' )

        #single decoder lstm
        self.decoder_lstm = LSTM(self.units, return_sequences=True, return_state=True, name = 'Decoder_LSTM')

        #decoder dropout layer
        self.decoder_dropout = Dropout(0.4, name = 'Decoder_Dropout')

        #TimeDistributed(Dense) layer
        self.decoder_dense =  TimeDistributed(Dense(y_size, activation='softmax'), name = 'Decoder_TimeDistributed')

    def forward(self):
        #Pass input through encoder layers
        encoder_outputs = self.encoder_embedding_layer(self.encoder_inputs)
        encoder_outputs = self.encoder_lstm1(encoder_outputs)
        encoder_outputs = self.encoder_lstm2(encoder_outputs)
        encoder_outputs = self.encoder_lstm3(encoder_outputs)


        #Pass input through decoder layers
        decoder_outputs = self.decoder_embedding_layer(self.decoder_inputs)
        decoder_outputs = self.decoder_lstm(decoder_outputs, initial_state=encoder_outputs[1:])[0]
        decoder_outputs = self.decoder_dropout(decoder_outputs)
        decoder_outputs = self.decoder_dense(decoder_outputs)


        self.model = Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)
        self.model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

        #Generate Inference Model
        self.encoder_model = Model(self.encoder_inputs, encoder_outputs)
        decoder_h = Input(shape=(self.units,))
        decoder_c = Input(shape=(self.units,))
        decoder_hidden = Input(shape=(self.max_text_len, self.units))

        decoder_embedding= self.decoder_embedding_layer(self.decoder_inputs) 
        decoder_inference_outputs, state_h, state_c = self.decoder_lstm(decoder_embedding, initial_state=[decoder_h, decoder_c])
        decoder_inference_outputs = self.decoder_dense(decoder_inference_outputs) 

        self.decoder_model = Model([self.decoder_inputs] + [decoder_hidden,decoder_h, decoder_c],
        [decoder_inference_outputs] + [state_h, state_c])

    def train(self, x_tr, y_tr, x_val, y_val):
        es = EarlyStopping(monitor='val_loss', mode='min', patience=1)
        history=self.model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,epochs=50,callbacks=[es],batch_size=64, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))
        return history

    def infer(self, input_seq):
        # Encode the input as state vectors
        index_word = self.y_tokenizer.index_word
        word_index = self.y_tokenizer.word_index

        e_out, e_h, e_c = self.encoder_model.predict(input_seq, verbose=0)
        
        # Set target sequence to start token
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = word_index['<s>']

        decoded_sentence = []
        sample = ''

        # if stop token is hit or sentence, stop running and return sentence
        while sample != '</s>' or len(decoded_sentence) == self.max_summary_len:
            output_tokens, e_h, e_c = self.decoder_model.predict([target_seq] + [e_out, e_h, e_c], verbose=0)

            # Sample top 3 tokens, pick a random token (Topk sampling)
            sample_index = np.random.choice(np.argpartition(output_tokens[0, -1, :],-3)[-3:])
        
            if sample_index != 0:
                sample = index_word[sample_index]

                if(sample!='</s>'):
                    decoded_sentence.append(sample) 

                # Update target sequence
                target_seq[0, 0] = sample_index

        return ' '.join(decoded_sentence)

    def predict(self, x_test):
        # predict summary from an article input
        return self.infer(x_test.reshape(1,self.max_text_len))

    def load_model(self,name):
        # load model, encoder inference model, decoder inference model
        self.model = keras.models.load_model(f"src/model{name}.h5")
        self.encoder_model = keras.models.load_model(f"src/encoder_model{name}.h5")
        self.decoder_model = keras.models.load_model(f"src/decoder_model{name}.h5")

    def save_model(self, name = ''):
        # save model parameters
        with open(f'src/parameters{name}.pickle', 'wb') as f:
            pickle.dump((self.units, self.output_dim, self.x_size, self.y_size, self.x_tokenizer, self.y_tokenizer, self.max_text_len, self.max_summary_len, self.embeddings), f)
        
        # save model, encoder inference model, decoder inference model
        self.model.save(f"src/model{name}.h5")
        self.encoder_model.save(f"src/encoder_model{name}.h5")
        self.decoder_model.save(f"src/decoder_model{name}.h5")
        
 








