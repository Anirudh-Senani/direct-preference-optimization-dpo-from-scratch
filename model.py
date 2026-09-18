"""
Direct Preference Optimization (DPO) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - log_softmax
def log_softmax(logits, axis=-1):
    # TODO: convert logits into numerically stable log-probabilities along axis
    shifted = logits - logits.max(axis=axis, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))

    return shifted - logsumexp

# Step 2 - softmax
def softmax(logits, axis=-1):
    # TODO: Convert an array of logits into a probability distribution along a given axis
    shifted = np.exp(logits - logits.max(axis=axis, keepdims=True))
    return shifted/shifted.sum(axis=axis, keepdims=True)

# Step 3 - gather_token_logprobs
def gather_token_logprobs(log_probs, token_ids):
    # TODO: Extract the log-probability of each observed token from a full vocab log-prob tensor...
    return log_probs[np.arange(token_ids.shape[0])[:,None], np.arange(token_ids.shape[1])[None,:], token_ids]

# Step 4 - masked_sequence_logprob
def masked_sequence_logprob(token_logprobs, mask):
    # TODO: Sum per-token log-probabilities under a binary mask to obtain a single sequence log-probability per example.
    return (token_logprobs * mask).sum(axis=-1)

# Step 5 - init_policy_params
def init_policy_params(vocab_size, d_model, rng=None):
    # TODO: Initialize the policy language-model parameters with small random values
    if rng is None:
        rng = np.random.default_rng()

    embed = rng.normal(loc=0.0, scale=0.02, size=(vocab_size, d_model))
    W_out = rng.normal(loc=0.0, scale=0.02, size=(d_model, vocab_size))
    b_out = np.zeros((vocab_size,))

    return dict(
        embed=embed,
        W_out=W_out,
        b_out=b_out
    )

# Step 6 - policy_token_logits
def policy_token_logits(params, token_ids):
    # TODO: Compute next-token logits for every position from policy params and token ids.
    feats = params['embed'][token_ids]
    return feats @ params['W_out'] + params['b_out']

# Step 7 - policy_sequence_logprob
def policy_sequence_logprob(params, token_ids, mask):
    # TODO: Compute the total masked sequence log-probability under the current policy...
    logits = policy_token_logits(params, token_ids)
    logprobs = log_softmax(logits)

    token_logprobs = gather_token_logprobs(logprobs, token_ids)
    return masked_sequence_logprob(token_logprobs, mask)

# Step 8 - sequence_logprob_grad
def sequence_logprob_grad(params, token_ids, mask):
    # TODO: Compute gradients of the summed sequence log-probability w.r.t. params
    logits = policy_token_logits(params, token_ids)
    x = params['embed'][token_ids]
    dlogits = -softmax(logits)

    dlogits[np.arange(token_ids.shape[0])[:,None], np.arange(token_ids.shape[1])[None, :], token_ids] += 1.0
    dlogits *= mask[..., np.newaxis]

    dx = dlogits @ params['W_out'].T
    dW = (x.transpose((0,2,1)) @ dlogits).sum(axis=0)
    db = dlogits.sum(axis=(0,1))
    dembed = np.zeros_like(params['embed'])

    np.add.at(dembed, token_ids, dx)

    return dict(
        embed=dembed,
        W_out=dW,
        b_out=db
    )

# Step 9 - bradley_terry_loss (not yet solved)
# TODO: implement

# Step 10 - reward_accuracy (not yet solved)
# TODO: implement

# Step 11 - build_preference_pairs (not yet solved)
# TODO: implement

# Step 12 - sample_preference_batch (not yet solved)
# TODO: implement

# Step 13 - freeze_reference_logprobs (not yet solved)
# TODO: implement

# Step 14 - policy_reference_logratio (not yet solved)
# TODO: implement

# Step 15 - dpo_pair_margin (not yet solved)
# TODO: implement

# Step 16 - dpo_loss (not yet solved)
# TODO: implement

# Step 17 - dpo_loss_grad (not yet solved)
# TODO: implement

# Step 18 - dpo_train_step (not yet solved)
# TODO: implement

# Step 19 - train_dpo (not yet solved)
# TODO: implement

# Step 20 - length_normalized_logprob (not yet solved)
# TODO: implement

# Step 21 - ipo_loss (not yet solved)
# TODO: implement

# Step 22 - implicit_reward (not yet solved)
# TODO: implement

# Step 23 - preference_accuracy (not yet solved)
# TODO: implement

# Step 24 - kl_to_reference (not yet solved)
# TODO: implement

# Step 25 - reward_margin_stats (not yet solved)
# TODO: implement

# Step 26 - evaluate_dpo (not yet solved)
# TODO: implement

# Step 27 - run_dpo_pipeline (not yet solved)
# TODO: implement

