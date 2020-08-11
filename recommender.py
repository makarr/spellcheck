import editdistance as ed
from nltk.corpus import words
from cmu_data import CMUData
from model_utils import load_model
from pronouncer import Pronouncer


class Recommender:
    
    def __init__(self):
        data = CMUData()
        model = load_model(data=data)
        self.data = data
        self.model = model
        self.pronouncer = Pronouncer(model, data)
        self.words = set(words.words())

    
    # returns list of words whose pronunciation is closest to that of `word`
    def recommend(self, word):
        least_dist = float('inf')
        best_match = []
        pron_dict = self.data.pron_dict
        predicted = self.pronouncer.pronounce(word)
        for pron in pron_dict.keys():
            dist = ed.eval(pron, predicted)
            if dist < least_dist:
                least_dist = dist
                best_match = pron_dict[pron]
            elif dist == least_dist:
                best_match += pron_dict[pron]
        matches = [''.join(match) for match in set(best_match)]
        filtered = [match for match in matches if match in self.words]
        if not filtered:
            filtered = matches
        return filtered