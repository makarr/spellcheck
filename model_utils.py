import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cmu_data import CMUData
from seq2seq import Seq2Seq



def load_model(path='seq2seq.pt', data=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if data is None:
        data = CMUData()
    model = Seq2Seq(dim = 128,
                    src_vocab_size = len(data.src_vocab),
                    tgt_vocab_size = len(data.tgt_vocab),
                    src_max_len = data.src_max_len + 2, # for sos, eos
                    tgt_max_len = data.tgt_max_len + 2,
                    pad_idx = data.pad_idx,
                    device = device).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model



def train_model():
    epochs = 25
    batch_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = 'seq2seq.pt'

    print('Getting data...')
    data = CMUData()
    train_dl, valid_dl = data.get_dataloaders(batch_size, device)

    print('Initializing model...')    
    model = Seq2Seq(dim = 128,
                    src_vocab_size = len(data.src_vocab),
                    tgt_vocab_size = len(data.tgt_vocab),
                    src_max_len = data.src_max_len + 2, # for sos, eos
                    tgt_max_len = data.tgt_max_len + 2,
                    pad_idx = data.pad_idx,
                    device = device).to(device)

    print(f'Model has {len(model)} parameters.')

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    print('Training...')
    fit(model, epochs, optimizer, train_dl, valid_dl, path, device)
    print('Training complete.')
    print(f'Returning best model, also saved to path: {path}')
    model.load_state_dict(torch.load(path))
    return model



def fit(model, epochs, optimizer, train_dl, valid_dl, path, device):
    best_acc = 0
    
    for epoch in range(epochs):
        losses, accs = [], []

        model.train()
        for batch in train_dl:
            # src = (N, src_len)
            # tgt = (N, tgt_len)
            src = batch.src.to(device)
            tgt = batch.tgt.to(device)

            # outs = (N, tgt_len-1)
            outs = model(src, tgt[:,:-1]) # exclude eos
            loss = get_loss(outs, tgt, model.pad_idx)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            for batch in valid_dl:
                src = batch.src.to(device)
                tgt = batch.tgt.to(device)

                outs = model(src, tgt[:,:-1])
                loss = get_loss(outs, tgt, model.pad_idx)
                acc = get_accuracy(outs, tgt, model.pad_idx)

                losses.append(loss)
                accs.append(acc)

        avg_loss = torch.stack(losses).mean().item()
        avg_acc = torch.stack(accs).mean().item()
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), path)
        print(f'Epoch: {epoch+1:>2}    Loss: {avg_loss:.3f}    Accuracy: {avg_acc:.3f}')



def get_loss(outs, tgt, pad_idx):
    return F.cross_entropy(
        outs.reshape(-1, outs.shape[-1]),
        tgt[:,1:].reshape(-1),
        ignore_index=pad_idx
    )



# ignore pad_idx
def get_accuracy(outs, tgt, pad_idx):
    tgt = tgt[:,1:]
    tgt_mask = (tgt == pad_idx)
    outs = torch.argmax(outs, dim=-1).masked_fill_(tgt_mask, pad_idx)
    all_matches = (outs == tgt).float().sum()
    all_toks = tgt_mask.numel() - tgt_mask.float().sum()
    return all_matches / all_toks