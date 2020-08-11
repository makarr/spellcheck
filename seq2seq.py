import torch
import torch.nn as nn

# Custom implementation of a vanilla seq2seq transformer, as introduced in
# Attention is All You Need. https://arxiv.org/abs/1706.03762
#
# This code borrows liberally from Ben Trevett and Aladdin Persson, with additional
# inspiration from HuggingFace and FastAI. See README.md for links.


class Seq2Seq(nn.Module):

    def __init__(self,
                 dim,            # dimension of attention layers
                 src_vocab_size, # used to determine token embeddings
                 tgt_vocab_size, # ...
                 src_max_len,    # used to determine positional embeddings
                 tgt_max_len,    # ...
                 pad_idx,        # the special index for pad tokens, used for loss / accuracy
                 device,         # the device hosting the computation
                 num_heads=8,    # number of heads per attention layer (divides into dim)
                 num_layers=3,   # number of encoder & decoder layers
                 expand=4,       # the factor by which to expand dim for internal attention layer
                 p=0.1):         # dropout probability
        super().__init__()
        self.encoder = Encoder(dim, src_vocab_size, src_max_len, num_heads, num_layers, expand, p)
        self.decoder = Decoder(dim, tgt_vocab_size, tgt_max_len, num_heads, num_layers, expand, p)
        self.fc_out = nn.Linear(dim, tgt_vocab_size)

        self.pad_idx = pad_idx
        self.device = device

        
    def __len__(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def make_src_mask(self, src):        
        # src = (N, len)
        # src_mask = (N, 1, 1, len)
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(src.device)

    
    def make_tgt_mask(self, tgt):     
        # tgt = (N, len)
        # tgt_pad_mask = (N, 1, 1, len)
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # tgt_sub_mask = (len, len)
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len)))
        tgt_sub_mask = tgt_sub_mask.bool().to(tgt.device)

        # tgt_mask = (N, 1, len, len)
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask

    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.fc_out(dec_out)



class Encoder(nn.Module):

    def __init__(self, dim, vocab_size, max_len, num_heads, num_layers, expand, p):
        super().__init__()
        self.embedding = Embedding(dim, vocab_size, max_len, p)
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, num_heads, expand, p)
             for _ in range(num_layers)]
        )

    
    def forward(self, src, mask=None):
        # src = (N, seq_len)
        out = self.embedding(src)
        for layer in self.layers:
            out = layer(out, out, out, mask)       
        # out = (N, seq_len, dim)
        return out



class Decoder(nn.Module):

    def __init__(self, dim, vocab_size, max_len, num_heads, num_layers, expand, p):
        super().__init__()
        self.embedding = Embedding(dim, vocab_size, max_len, p)
        self.layers = nn.ModuleList(
            [DecoderBlock(dim, num_heads, expand, p) for _ in range(num_layers)]
        )

    
    def forward(self, tgt, enc_out, src_mask, tgt_mask):
        # tgt = (N, len)
        # enc_out = (N, len, dim)
        
        # tgt_out = (N, len, dim)
        tgt_out = self.embedding(tgt)
        for layer in self.layers:
            tgt_out = layer(tgt_out, enc_out, enc_out, src_mask, tgt_mask)
        
        return tgt_out



class DecoderBlock(nn.Module):
    
    def __init__(self, dim, num_heads, expand, p):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, num_heads)
        self.transformer_block = TransformerBlock(dim, num_heads, expand, p)
        self.dropout = nn.Dropout(p)

    
    def forward(self, x, key, value, src_mask=None, tgt_mask=None):
        attention = self.attention(x, x, x, tgt_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(query, key, value, src_mask)
        return out



class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, expand, p):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, expand*dim),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(expand*dim, dim)
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p)
    
    
    def forward(self, query, key, value, mask=None):
        # x = (N, len, dim)
        # mask = (N, len)

        # attn_out = (N, len, dim)
        attn_out = self.attention(query, key, value, mask)
        attn_out = self.dropout(attn_out)
        attn_out = self.attn_norm(query + attn_out)

        # ff_out = (N, len, dim)
        ff_out = self.ff(attn_out)
        ff_out = self.dropout(ff_out)
        ff_out = self.ff_norm(attn_out + ff_out)

        return ff_out



class MultiHeadAttention(nn.Module):

    def __init__(self, dim, num_heads):
        assert dim % num_heads == 0
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        scale = torch.rsqrt(torch.tensor(dim).float())
        self.scale = nn.Parameter(scale, requires_grad=False)

        self.Q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.fc_out = nn.Linear(dim, dim)


    def forward(self, query, key, value, mask=None):
        # q,k,v: (N, len, dim)
        N = query.shape[0]
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]

        query = self.Q(query)
        key = self.K(key)
        value = self.V(value)

        # q,k,v: (N, len, num_heads, head_dim)
        query = query.reshape(N, query_len, self.num_heads, self.head_dim)
        key = key.reshape(N, key_len, self.num_heads, self.head_dim)
        value = value.reshape(N, value_len, self.num_heads, self.head_dim)

        # energy: (N, num_heads, query_len, key_len)
        energy = torch.einsum('nqhd,nkhd->nhqk', [query, key]) * self.scale # really dividing, but used rsqrt

        if mask is not None:
            energy.masked_fill_(mask == 0, float('-inf'))

        # attention: (N, heads, query_len, key_len)
        attention = torch.softmax(energy, dim=-1)

        # out: (N, query_len, num_heads, head_dim)
        out = torch.einsum('nhql,nlhd->nqhd', [attention, value])
        
        # out: (N, query_len, dim)
        out = out.reshape(N, query_len, self.dim)
        out = self.fc_out(out)
        return out



class Embedding(nn.Module):

    def __init__(self, dim, vocab_size, max_len, p):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_len, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p)
        scale = torch.sqrt(torch.tensor(dim).float())
        self.scale = nn.Parameter(scale, requires_grad=False)


    def forward(self, seq):
        # seq = (N, len)
        N, seq_len = seq.shape
        pos = torch.arange(0, seq_len).unsqueeze(0).expand(N, seq_len).to(seq.device)

        # embeds = (N, len, dim)
        embeds = self.tok_embedding(seq) * self.scale + self.pos_embedding(pos)
        return self.dropout(self.norm(embeds))