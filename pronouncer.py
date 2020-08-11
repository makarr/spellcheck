import torch

# translates alphabet sequences into phoneme sequences
class Pronouncer:
    
    def __init__(self, model, data):
        self.model = model
        self.src_vocab = data.src_vocab
        self.tgt_vocab = data.tgt_vocab
        self.src_max_len = data.src_max_len
        self.tgt_max_len = data.tgt_max_len

    
    # modified from translate_sentence() here:
    # https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
    def pronounce(self, word):
        if len(word) > self.src_max_len:
            print(f'{word} is length {len(word)}, which exceeds max length {self.src_max_len}')
            exit()

        model = self.model
        device = model.device
        
        word_toks = ([self.src_vocab.stoi['<sos>']] + 
                     [self.src_vocab.stoi[char] for char in word] + 
                     [self.src_vocab.stoi['<eos>']])
        word_tensor = torch.tensor(word_toks).unsqueeze(0).to(device)

        outputs = [self.tgt_vocab.stoi['<sos>']]
        
        model.eval()
        for i in range(self.tgt_max_len):
            pron_tensor = torch.tensor(outputs).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(word_tensor, pron_tensor)

            best_guess = output.argmax(-1)[:,-1].item()
            outputs.append(best_guess)

            if best_guess == self.tgt_vocab.stoi['<eos>']:
                break

        pron_toks = [self.tgt_vocab.itos[idx] for idx in outputs]
        pron_toks = pron_toks[1:] # remove sos token
        if pron_toks[-1] == '<eos>':
            pron_toks = pron_toks[:-1] # remove eos token
        
        return tuple(pron_toks)