class SelfAttention(nn.Module):
    def __init__(self, din, dout, qkv_bias=False):
        super().__init__()
        self.W_query=nn.Linear(din, dout, bias=qkv_bias)
        self.W_key=nn.Linear(din, dout, bias=qkv_bias)
        self.W_value=nn.Linear(din, dout, bias=qkv_bias)
    def forward(self, x):
        queries=self.W_query(x)
        key=self.W_key(x)
        value=self.W_value(x)

        attn_scores=queries @ key.T

        attn_wt=torch.softmax(attn_scores/key.shape[-1]**0.5, dim=-1)

        context_vectors=attn_wt @ value
        return context_vectors, attn_scores, attn_wt, key



torch.manual_seed(789)

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your   (x^1)
     [0.55, 0.87, 0.66], # journey (x^2)
     [0.57, 0.85, 0.64], # starts  (x^3)
     [0.22, 0.58, 0.33], # with    (x^4)
     [0.77, 0.25, 0.10], # one     (x^5)
     [0.05, 0.80, 0.55]] # step    (x^6)
)
din=3
dout=3
self_att=SelfAttention(din, dout)
context_vectors, attn_scores, attn_wt, key=self_att(inputs)
