from model import load_parameters
from model import load_ds
from model import TextSummarizerModel
from cosine_sim import summarize
from evaluation_tools import average_rouge_bleu
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import backend as K 

n = [1000, 5000, 10000, 50000]

for i in n:
    # load in parameters and data from pickle files
    units, output_dim, x_size, y_size, x_tokenizer, y_tokenizer, max_text_len, max_summary_len, embeddings = load_parameters(name=i)
    unclean_x, unclean_y, clean_x, clean_y, x_seq = load_ds(name = i)

    # initialize encoder-decoder model and load in parameters
    model = TextSummarizerModel(units, output_dim, x_size, y_size, x_tokenizer, y_tokenizer, max_text_len, max_summary_len, embeddings)
    model.load_model(i)

    # generate articles using extractive and abstractive models and calculate average ROUGE and BLEU scores
    predictions = []
    predictions_ext = []
    references = unclean_y
    originals = unclean_x
    x_seq = pad_sequences(x_tokenizer.texts_to_sequences(unclean_x), maxlen=max_text_len, padding='post')
    for j in range(len(unclean_x)):
        K.clear_session()
        predictions.append(model.predict(x_seq[j]))
    print("Abstractive Text Summarization----------------------------------------------------",flush=True)
    average_rouge_bleu(predictions, references)
    print('',flush=True)
    print("Extractive Text Summarization-----------------------------------------------------",flush=True)
    for k in range(len(unclean_x)):
        predictions_ext.append(summarize(unclean_x[k]))
    average_rouge_bleu(predictions_ext, references)