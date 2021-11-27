from utils import load
import numpy as np

from nltk.translate.bleu_score import sentence_bleu

references = [[[1, 2, 3, 4]], [[1, 3, 5, 2]]]
hypotheses = [[1, 2, 4, 5], [6, 3, 5, 2]]



_dict = load("./objects/processed_captions_training.pkl")
word_map = _dict["word_map"]
rev_word_map = {v: k for k, v in word_map.items()}

ref_sentences = []
for item in references:
    ref = item[0]
    words = [ rev_word_map[word] for word in ref ]
    sentence = list(" ".join(words))
    ref_sentences.append(sentence)

hyp_sentenses = []
for item in hypotheses:
    words = [ rev_word_map[word] for word in item ]
    sentence = list(" ".join(words))
    hyp_sentenses.append(sentence)

score = np.mean([sentence_bleu([list(ref_sentences[i])], list(hyp_sentenses[i])) for i in range(len(hyp_sentenses))])
print(score)