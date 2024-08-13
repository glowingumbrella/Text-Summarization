from rouge_score import rouge_scorer
import evaluate
from nltk.translate.bleu_score import sentence_bleu

#get rouge score
def rouge_score(prediction, reference, component = False):
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=[prediction],references=[reference])
    if component:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
        scores = scorer.score(prediction, reference)
        return results, scores
    return results

#get bleu score
def bleu_score(prediction, reference, bigram = False):
    prediction = prediction.split()
    reference = [reference.split()]
    if bigram == True:
        score = sentence_bleu(reference, prediction, weights=(0,1,0,0))
        return score
    score = sentence_bleu(reference, prediction, weights=(1,0,0,0))
    return score

#get average rouge and bleu scores
def average_rouge_bleu(prediction, reference):
    n = len(prediction)
    rouge1 = 0
    rouge2 = 0
    rougeL = 0
    rougeLsum = 0
    bleu_uni = 0
    bleu_bi = 0
    for i in range(n):
        rouge = rouge_score(prediction[i],reference[i])
        rouge1 += rouge['rouge1']
        rouge2+= rouge['rouge2']
        rougeL += rouge['rougeL']
        rougeLsum += rouge['rougeLsum']
        bleu_uni += bleu_score(prediction[i],reference[i])
        bleu_bi += bleu_score(prediction[i],reference[i],bigram = True)
    print(f"{n} tests",flush=True)
    print("--------------------------",flush=True)
    print('rouge1:',rouge1/n,flush=True)
    print('rouge2:',rouge2/n,flush=True)
    print( 'rougeL:',rougeL/n,flush=True)
    print('rougeLsum:',rougeLsum/n,flush=True)
    print( 'bleu unigram:',bleu_uni/n,flush=True)
    print( 'bleu bigram:',bleu_bi/n,flush=True)
    print('',flush=True)

