from recommender import Recommender

def main():
   r = Recommender()

   print("Enter 'q' to quit.\n")

   while True:
      word = input('Enter a word: ')
      if word == 'q': exit()
      recs = r.recommend(word)
      print(f'Did you mean: {recs[0]}')
      if len(recs) > 1:
         for rec in recs[1:]:
            print(' '*14 + rec)
      print()

if __name__ == '__main__':
   main()