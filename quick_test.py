import editdistance as ed
from recommender import Recommender

words = [
   'polieyerathang',
   'akshulie',
   'hoapflea',
   'kuzz-zyn',
   'zookeeny',
   'seprit',
   'apsirduhtee',
   'enphrakshun',
   'real-eyes',
]

r = Recommender()
p = r.pronouncer
word_dict = r.data.word_dict

print()
for word in words:
   rec = r.recommend(word)[0]
   print(f'Word:              {word}')
   print(f'Recommendation:    {rec}')
   print(f'Lexical distance:  {ed.eval(word, rec)}')
   print(f'Phonemic distance: {ed.eval(p.pronounce(word), word_dict[tuple(rec)][0])}\n')