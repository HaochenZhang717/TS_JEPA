# Plan: DTS-JEPD

## Goal

Build a time-series conditional generation pretraining variant on top of TS-JEPA.

The model should preserve the original TS-JEPA representation objective while adding a weak patch-level conditional generation objective inspired by D-JEPA and JiT. The conditional generation target is each masked raw patch. The condition is the predictor output for the corresponding masked patch.

Working name: DTS-JEPD, meaning Denoising Time-Series JEPA with patch-level denoising.

## Core Idea

For each batch:

1. Load multivariate or univariate time-series patches.
2. Randomly mask patch indices.
3. Encode visible patches with the context encoder.
4. Use the predictor to produce predicted latent embeddings for masked patches.
5. Compute the original JEPA latent prediction loss against EMA encoder targets.
6. Feed the clean masked raw patches, JiT noising timestep, and predictor embeddings into a small denoising head.
7. Compute JiT-style velocity loss.
8. Optimize:

```text
L_total = L_jepa + lambda_denoise * L_denoise
```

## Alignment With Existing TS-JEPA

Keep these parts aligned with `pretrain_multivariate.py`:

- Patch construction: `[B, N, P, C]` for multivariate input.
- Masking at patch level.
- Context encoder sees only non-masked patches.
- EMA encoder sees all patches, then masked target embeddings are selected.
- Predictor receives visible embeddings and predicts embeddings for masked positions.
- JEPA loss is computed only on masked predicted embeddings.
- EMA target encoder is updated by moving average of context encoder.
- AdamW optimizer and current W&B logging style.

## Difference From Existing TS-JEPA

Add a small conditional denoising head after predictor output:

```text
predicted masked embedding z_m: [B, M, D]
clean masked patch x:           [B, M, P, C]
JiT timestep t:                 [B, M] or [B*M]
noisy patch z_t:                [B, M, P, C]

denoising head(z_t, t, z_m) -> x_pred: [B, M, P, C]
```

The denoising loss should be computed only on masked patches.

## JiT-Style Formulation

Use the formulation from the reference JiT code:

```text
t ~ sigmoid(N(P_mean, P_std))
e ~ N(0, noise_scale^2 I)
z_t = t * x + (1 - t) * e
v_target = (x - z_t) / clamp(1 - t, min=t_eps)
x_pred = denoising_head(z_t, t, z_m)
v_pred = (x_pred - z_t) / clamp(1 - t, min=t_eps)
L_denoise = mean((v_target - v_pred)^2)
```

Important details:

- This is an `x` prediction parameterization with velocity loss.
- The path starts from noise near `t=0` and moves toward data near `t=1`.
- Use `clamp(1 - t, min=t_eps)` for numerical stability.
- `x_mask` from the JiT image code is conceptually replaced by selecting only masked patches. Since all selected masked patches are valid, an explicit spatial mask is not needed unless variable-length patches are introduced later.

## Denoising Head Design

The head should remain intentionally weak.

Recommended first version:

```text
z_t: [B, M, P, C]
reshape -> [B*M, C, P]

Conv1d(C -> hidden, kernel_size=3, padding=1)
AdaLN conditioning from time embedding and predictor embedding
activation
Conv1d(hidden -> C, kernel_size=1) or Linear(hidden*P -> P*C)
reshape -> [B, M, P, C]
```

The user requested a simple backbone: one CNN layer, one activation layer, one Linear layer.

A compatible implementation can be:

```text
Conv1d(C -> hidden, kernel_size=3, padding=1)
AdaLN(hidden, condition)
GELU
flatten over hidden and patch length
Linear(hidden * P -> P * C)
```

Condition injection should follow the JiT-style conditioning pattern:

```text
t_emb = TimestepEmbedder(hidden)(t)
z_emb = Linear(predictor_embed_dim -> hidden)(z_m)
c = t_emb + z_emb
```

Then use AdaLN to modulate the normalized convolution features:

```text
shift, scale = adaLN_modulation(c).chunk(2, dim=-1)
features = norm(features)
features = features * (1 + scale[..., None]) + shift[..., None]
```

For the simple CNN head, `features` will have shape:

```text
[B*M, hidden, patch_size]
```

and `c` will have shape:

```text
[B*M, hidden]
```

This mirrors JiT's core pattern:

```text
t_emb = self.t_embedder(t)
label_emb = self.label_embedder(label)
c = t_emb + label_emb
```

but replaces class-label conditioning with predictor-output conditioning:

```text
c = t_emb + predictor_cond_emb
```

For the first implementation, avoid a deep denoising network. The purpose is to add weak patch-level conditional generation pressure, not to replace the JEPA objective.

## Time Embedding

Use a JiT-style sinusoidal timestep embedding followed by an MLP.

Recommended:

```text
TimestepEmbedder(hidden_size, frequency_embedding_size)
t_emb = TimestepEmbedder(t) -> [B*M, hidden]
z_emb = Linear(predictor_embed_dim -> hidden)(z_m) -> [B*M, hidden]
c = t_emb + z_emb
```

Keep `frequency_embedding_size` modest, e.g. 64 or 128.

This is deliberately additive conditioning, not concatenation, to stay close to JiT's `time + label` design.

## AdaLN Details

Use AdaLN as the conditioning mechanism for the denoising head.

For a one-layer CNN head:

```text
conv_features = Conv1d(z_t)
normed = LayerNorm/RMSNorm over hidden channels
shift, scale = Linear(SiLU(c)) -> 2 * hidden
modulated = normed * (1 + scale[..., None]) + shift[..., None]
activated = GELU(modulated)
out = Linear(flatten(activated)) -> patch_size * channels
```

Because `Conv1d` uses `[B*M, hidden, patch_size]`, either:

```text
transpose to [B*M, patch_size, hidden] before LayerNorm
```

or use a channel-wise normalization that supports `[B*M, hidden, patch_size]`.

Recommended first implementation:

```text
conv_features: [B*M, hidden, P]
conv_features = conv_features.transpose(1, 2)  # [B*M, P, hidden]
normed = LayerNorm(hidden)(conv_features)
shift, scale = adaLN(c).chunk(2, dim=-1)
modulated = normed * (1 + scale[:, None, :]) + shift[:, None, :]
activated = GELU(modulated)
flatten -> Linear(P * hidden -> P * C)
```

Zero-initialize the final AdaLN modulation layer if possible, following JiT/DiT practice. Also consider zero-initializing the final output linear layer so the denoising head starts conservatively.

## Loss Weighting

Expose:

```text
--lambda_denoise
```

Start with:

```text
0.001, 0.01, 0.05, 0.1
```

Default recommendation:

```text
lambda_denoise = 0.01
```

The denoising loss is dense and can dominate. Keep JEPA as the primary representation objective.

## Config / CLI Parameters

Add arguments:

```text
--lambda_denoise
--denoise_hidden_dim
--time_embed_dim
--time_frequency_dim
--P_mean
--P_std
--t_eps
--noise_scale
```

Suggested defaults:

```text
lambda_denoise = 0.01
denoise_hidden_dim = 128
time_embed_dim = 128
time_frequency_dim = 128
P_mean = 0.0
P_std = 1.0
t_eps = 1e-5
noise_scale = 1.0
```

## Logging

Log the following to W&B:

```text
train/jepa_loss
train/denoise_loss
train/total_loss
train/lr
train/ema_momentum
train/lambda_denoise
data/num_channels
```

Also include denoising hyperparameters in run names:

```text
lambda_denoise
denoise_hidden_dim
P_mean
P_std
noise_scale
time_embed_dim
time_frequency_dim
```

## Checkpoint Contents

Save:

```text
context encoder state_dict
predictor state_dict
denoising head state_dict
epoch
selected_cols
args/config
```

The original `pretrain.py` saves only encoder. For DTS-JEPD, generation needs predictor plus denoising head, so save all trainable generation components.

## Sampling Design

Sampling does not need to be implemented in the first pass, but training should make sampling straightforward.

Given context patches and masked target positions:

1. Encode visible context.
2. Predict masked embeddings `z_m`.
3. Initialize masked patches from noise:

```text
z_0 = noise_scale * N(0, I)
```

4. Integrate from `t=0` to `t=1` using Euler first:

```text
x_pred = denoising_head(z_t, t, z_m)
v_pred = (x_pred - z_t) / clamp(1 - t, min=t_eps)
z_next = z_t + (t_next - t) * v_pred
```

5. Optionally add Heun later if Euler works.

## Project Structure

Implement DTS-JEPD in a new self-contained path so the original repo remains clean and the baseline files stay untouched.

Recommended path:

```text
dtsjepd/
```

Required files:

```text
dtsjepd/models.py
dtsjepd/datasets.py
dtsjepd/train.py
```

Optional later files:

```text
dtsjepd/eval.py
dtsjepd/sampling.py
dtsjepd/utils.py
```

The first implementation should use only the three required files unless shared helper code becomes clearly useful.

## File Responsibilities

`dtsjepd/models.py`

Contains all model definitions:

```text
MultivariateTSTokenizer
MultivariateEncoder
TimestepEmbedder
AdaLN patch denoising head
DTSJEPD wrapper module if useful
```

It may reuse existing repo modules when appropriate:

```text
src.models.predictor.Predictor
src.models.utils.modules.Block
src.models.utils.mask_utils.apply_mask
```

`dtsjepd/datasets.py`

Contains dataset and dataloader logic:

```text
MultivariateCSVDataLoader
column selection logic
train/val/test split
train-only normalization
masked patch sampling
helper function build_dataloaders(...)
```

It must support both:

```text
data/weather/weather.csv
data/household/household.csv
```

Default column behavior:

```text
weather   -> all numeric columns
household -> original 0..5 columns, not duplicated OT
```

`dtsjepd/train.py`

Contains:

```text
argument parsing
model construction
optimizer/scheduler construction
training loop
validation/evaluation loop
checkpoint saving
W&B initialization/logging
smoke-test friendly CLI args
```

The file should be runnable as:

```text
python -m dtsjepd.train --data household ...
```

or:

```text
python dtsjepd/train.py --data household ...
```

Prefer supporting both if import paths remain simple.

## Training And Evaluation Loops

Training loop:

```text
for epoch:
    train on train split
    log train statistics
    periodically run evaluation on validation split
    save checkpoint
```

Evaluation loop:

Use the same losses as training, but no optimizer step:

```text
val/jepa_loss
val/denoise_loss
val/total_loss
```

Evaluation should use validation data only. Test data should stay untouched unless the user explicitly asks for final testing.

Initial evaluation frequency:

```text
--eval_every 10
```

Also expose:

```text
--disable_eval
```

for quick debugging runs.

Validation masking can remain random at first, matching training. Later, add a fixed validation mask option if curves are too noisy.

## W&B Monitoring

During training, W&B should monitor both training and evaluation statistics.

Required train logs:

```text
train/jepa_loss
train/denoise_loss
train/total_loss
train/weighted_denoise_loss
train/lr
train/ema_momentum
train/lambda_denoise
train/epoch_time_sec
data/num_channels
```

Required validation logs:

```text
val/jepa_loss
val/denoise_loss
val/total_loss
val/weighted_denoise_loss
```

Optional useful logs:

```text
train/denoise_to_jepa_ratio
val/denoise_to_jepa_ratio
train/pred_embedding_norm
train/target_embedding_norm
train/denoise_condition_norm
```

W&B config must include:

```text
all CLI args
selected columns
number of channels
patch size
number of patches
model dimensions
JiT noising parameters
```

Run names should include:

```text
data
num_channels
lr
batch size
ema momentum
mask ratio
lambda_denoise
denoise_hidden_dim
encoder dimensions
predictor dimensions
```

## Implementation Recommendation

Implement the new module from `pretrain_multivariate.py` concepts, but do not keep it as a single monolithic script.

Suggested implementation order:

1. Create `dtsjepd/datasets.py` with train/val/test loaders.
2. Create `dtsjepd/models.py` with encoder, timestep embedder, AdaLN denoising head.
3. Create `dtsjepd/train.py` with JEPA loss, JiT denoising loss, training, validation, W&B, and checkpointing.
4. Add checkpoint saving for encoder, predictor, denoising head, selected columns, and config.
5. Run smoke tests:

```text
household, num_epochs=1
weather, num_epochs=1
```

## Risks And Things To Watch

The denoising loss may overpower JEPA. Watch the ratio of:

```text
lambda_denoise * L_denoise
```

to:

```text
L_jepa
```

If representation quality degrades, reduce `lambda_denoise`.

The head may ignore predictor condition if the noisy patch already contains too much information near `t=1`. JiT's timestep distribution helps, but inspect performance across different `t` bins if needed.

If the denoising head is too weak, the loss may not help. Increase `denoise_hidden_dim` before adding depth.

If generation samples are too smooth, move from weak head to a slightly stronger residual CNN or add direct `v` prediction as an ablation.

## Open Design Questions

1. Should the denoising loss backpropagate into encoder and predictor?

Initial answer: yes, with small `lambda_denoise`.

2. Should predictor output be detached before denoising?

Initial answer: no for main DTS-JEPD. Add `--detach_denoise_condition` only as an ablation if needed.

3. Should denoising use masked patches only?

Initial answer: yes. This is aligned with both TS-JEPA and D-JEPA.

4. Should we train univariate and multivariate variants?

Initial answer: build on multivariate path. It can still run univariate if selected columns contain one channel.
