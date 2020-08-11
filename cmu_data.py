from torchtext.data import Field, BucketIterator, Dataset, Example, interleave_keys
from random import shuffle
from os import path

URL = 'http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b'
PATH = 'cmu_dict.txt'
ALPHABET = set("abcdefghijklmnopqrstuvwxyz'.-")


# all-in-one container for the necessary data. used for training and inference.
# TODO: optimize for different use cases
class CMUData:  
    def __init__(self):
        if not path.exists(PATH): download_cmudict()
        entries, word_dict, pron_dict, word_max_len, pron_max_len = load_cmudict()       
        SRC = Field(init_token = '<sos>',
                    eos_token = '<eos>',
                    batch_first = True)
        TGT = Field(init_token = '<sos>',
                    eos_token = '<eos>',
                    batch_first = True)
        fields = (('src', SRC), ('tgt', TGT))
        shuffle(entries)
        examples = [Example.fromlist(entry, fields) for entry in entries]
        dataset = TranslationDataset(examples, fields)
        SRC.build_vocab(dataset)
        TGT.build_vocab(dataset)
        self.dataset = dataset
        self.word_dict = word_dict
        self.pron_dict = pron_dict
        self.src_vocab = SRC.vocab
        self.tgt_vocab = TGT.vocab
        self.src_max_len = word_max_len
        self.tgt_max_len = pron_max_len
        self.pad_idx = TGT.vocab.stoi['<pad>']


    def get_dataloaders(self, batch_size, device):
    	train_data, valid_data = self.dataset.split()
    	train_dl, valid_dl = BucketIterator.splits(
        	(train_data, valid_data),
        	batch_size = batch_size,
        	device = device
        )
    	return train_dl, valid_dl



class TranslationDataset(Dataset):
    def __init__(self, examples, fields):
        super().__init__(examples, fields)

    
    # called by BucketIterator
    @staticmethod
    def sort_key(ex):
        return interleave_keys(len(ex.src), len(ex.tgt))



# loads from disk and preprocesses tokens
# returns a list of tuple, two dictionaries, and two ints
# TODO: save the preprocessed data to a new file
def load_cmudict():
    entries, word_dict, pron_dict = [], {}, {}
    word_max_len, pron_max_len = 0, 0
    with open(PATH, 'r', encoding='latin-1') as f:
        for line in f.readlines():
            if not line[0].isalpha(): # filter out heading and symbol pronunciations
                continue
            word, pron = line.split('  ')
            word = process_word(word)
            if word:
                pron = process_pron(pron)
                
                if len(word) > word_max_len:
                    word_max_len = len(word)                
                if len(pron) > pron_max_len:
                    pron_max_len = len(pron)

                if word in word_dict:
                    word_dict[word] += [pron]
                else:
                    word_dict[word] = [pron]

                if pron in pron_dict:
                    pron_dict[pron] += [word]
                else:
                    pron_dict[pron] = [word]
                
                entries.append((word, pron))
    return entries, word_dict, pron_dict, word_max_len, pron_max_len



def process_word(word):
    word = word.lower()
    # remove '(1)', '(2)', etc., duplicates tags 
    if word[-1] == ')':
        word = word[:-3]
    # filter out rare characters
    for char in word:
        if char not in ALPHABET:
            return None
    return tuple(word)



def process_pron(pron):
    pron = pron.split(' ')
    pron[-1] = pron[-1][:-1] # remove '\n'
    for i, phoneme in enumerate(pron):
        # remove syllabic stress
        if phoneme[-1].isdigit():
            phoneme = phoneme[:-1]
            pron[i] = phoneme
    return tuple(pron)



def download_cmudict():
	import requests
	r = requests.get(URL)
	with open(PATH, 'wb') as f:
		f.write(r.content)