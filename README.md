# LanguageModels

### KV Cache

Without a cache, `generate()` re-runs the full forward pass over the *entire* sequence generated so far, every single step — recomputing K and V for tokens whose values haven't changed since the last step. KV-cache is the technique of storing K and V the first time they're computed and reusing them, so each new step only computes Q, K, V for the one new token.

Since each prediction only depends on the *last* token's query attending over all keys so far, as this attention-score matrix shows:

```
          k_A    k_B    k_C
q_A  [  qA·kA   -inf   -inf ]
q_B  [  qB·kA  qB·kB   -inf ]
q_C  [  qC·kA  qC·kB  qC·kC ]
```

— a token's K and V get reused in every future step's row (e.g. `k_A` appears in every row), but its Q only ever appears in its own row, which is discarded the moment a later token becomes "current." That's why we cache K and V, but not Q.

### Implementation steps

1. **`Head.forward`** — accepts an optional `past_kv`. If present, extract the past K/V and concatenate them with the newly computed K/V for the current token(s) along the time dimension. The causal mask (`self.tril`) slice also has to account for the new total sequence length (`T_past` to `T_total`), not just the new tokens' length.
2. **`MultiHeadAttention.forward`** — accepts a `past_kv` list (or `None`). Loops over each head, passing head `i` its own `past_kv[i]` (or `None` on the first call), and collects each head's updated cache into a new list to return.
3. **`Block.forward`** — passes `past_kv` through to `self.sa` (the attention layer) and returns the updated cache alongside the residual output.
4. **`BigramLanguageModel`** — `self.blocks` had to change from `nn.Sequential` to `nn.ModuleList`, since `Sequential` auto-chains a single input through each submodule and can't pass a different `past_kv[i]` to each of the 4 blocks or collect per-block caches back out. `forward` now loops over the blocks manually. A new `generate_with_cache` method feeds the full prompt on the first step (no cache yet), then only the single newest token on every step after — relying on the cache for everything before it.

### Results

Measured generating tokens from an empty context, comparing `generate()` (no cache) vs `generate_with_cache()`, on a small CPU-scale model (n_embd=64, n_head=4, n_layer=4, block_size=64):

| | tokens | time | tok/s |
|---|---|---|---|
| No cache | 64 | 0.129s | ~495 |
| With cache | 64 | 0.089s | ~719 |

~31% faster even at this short length. Theory predicts the gap should widen further at longer sequences, since no-cache cost grows per step while cached cost stays roughly constant — not yet verified at longer lengths here (see limitations below).

### Known limitations / what's next

TODO: fill in — batching (not implemented), the `block_size` ceiling we hit (generation currently breaks past `block_size` tokens since `self.tril` and the position embedding table are both fixed-size, and the cache has no eviction/sliding-window logic), and PagedAttention as the natural next step for memory-efficient cache management at scale.
