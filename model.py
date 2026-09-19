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

# Step 9 - bradley_terry_loss
def bradley_terry_loss(reward_chosen, reward_rejected):
    # TODO: Compute the mean Bradley-Terry pairwise preference loss...
    reward_margin = reward_chosen - reward_rejected
    log_sigmoid = np.log(1/(1+np.exp(-reward_margin)))

    return (-log_sigmoid).mean()

# Step 10 - reward_accuracy
def reward_accuracy(reward_chosen, reward_rejected):
    # TODO: Fraction of pairs where chosen reward is strictly higher than rejected.
    return (reward_chosen > reward_rejected).mean()

# Step 11 - build_preference_pairs
def build_preference_pairs(prompts, chosen_ids, rejected_ids, chosen_mask, rejected_mask):
    # TODO: Package raw arrays into a list of preference-pair dictionaries
    pairs = []
    for i in range(len(prompts)):
        pair = {}
        pair['prompt'] = np.array(prompts[i])
        pair['chosen_ids'] = chosen_ids[i]
        pair['rejected_ids'] = rejected_ids[i]
        pair['chosen_mask'] = chosen_mask[i]
        pair['rejected_mask'] = rejected_mask[i]

        pairs.append(pair)

    return pairs

# Step 12 - sample_preference_batch
def sample_preference_batch(pairs, batch_size, rng=None):
    # TODO: Sample a mini-batch of preference pairs for one training step.
    if rng is None:
        rng = np.random.default_rng()

    n = len(pairs)
    replace = False
    if batch_size > n:
        replace = True

    inds = rng.choice(n, size=batch_size, replace=replace).tolist()

    batch = {}
    for key in pairs[0]:
        batch[key] = []

    for ind in inds:
        for key in batch:
            batch[key].append(pairs[ind][key])

    for key in batch:
        batch[key] = np.stack(batch[key])

    return batch

# Step 13 - freeze_reference_logprobs
def freeze_reference_logprobs(ref_params, pairs):
    # TODO: Precompute and freeze reference-model sequence log-probabilities for every chosen and rejected response...
    ref_probs = []
    for pair in pairs:
        ref_prob = {}
        ref_prob['chosen'] = policy_sequence_logprob(ref_params, pair['chosen_ids'][None,:], pair['chosen_mask'])
        ref_prob['rejected'] = policy_sequence_logprob(ref_params, pair['rejected_ids'][None,:], pair['rejected_mask'])

        ref_prob['chosen'] = float(np.asarray(ref_prob['chosen']).reshape(-1)[0])
        ref_prob['rejected'] = float(np.asarray(ref_prob['rejected']).reshape(-1)[0])

        ref_probs.append(ref_prob)

    return ref_probs

# Step 14 - policy_reference_logratio
def policy_reference_logratio(policy_logprob, reference_logprob):
    # TODO: Compute the per-sequence log-ratio log pi_theta(y) - log pi_ref(y)
    return policy_logprob - reference_logprob

# Step 15 - dpo_pair_margin
def dpo_pair_margin(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # TODO: Compute the scaled DPO pair margin for a batch of preference pairs
    return beta*(policy_reference_logratio(policy_logprob_chosen, ref_logprob_chosen) - policy_reference_logratio(policy_logprob_rejected, ref_logprob_rejected))

# Step 16 - dpo_loss
def dpo_loss(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # TODO: return the mean logistic loss on the DPO pair margins as a scalar float
    margin = dpo_pair_margin(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta)
    return (-np.log(1/(1 + np.exp(-margin)))).mean()

# Step 17 - dpo_loss_grad
def dpo_loss_grad(params, batch, ref_logprobs_batch, beta):
    # TODO: Evaluate DPO loss and return parameter gradients for the policy
    policy_logprobs_chosen = policy_sequence_logprob(params, batch['chosen_ids'], batch['chosen_mask'])
    policy_logprobs_rejected = policy_sequence_logprob(params, batch['rejected_ids'], batch['rejected_mask'])

    margin = dpo_pair_margin(policy_logprobs_chosen, policy_logprobs_rejected, ref_logprobs_batch['chosen'], ref_logprobs_batch['rejected'], beta)
    loss = np.logaddexp(0.0, -margin).mean()

    w = -(1/(1+np.exp(margin)))*beta/batch['chosen_ids'].shape[0]
    grads = {}
    for key in params:
        grads[key] = np.zeros_like(params[key])

    for i in range(batch['chosen_ids'].shape[0]):
        chosen_grads = sequence_logprob_grad(params, batch['chosen_ids'][i:i+1], batch['chosen_mask'][i:i+1])
        rejected_grads = sequence_logprob_grad(params, batch['rejected_ids'][i:i+1], batch['rejected_mask'][i:i+1])

        for key in grads:
            grads[key] += w[i] * (chosen_grads[key] - rejected_grads[key])

    return float(loss), grads

# Step 18 - dpo_train_step
import numpy as np

def dpo_train_step(params, batch, ref_logprobs_batch, beta, learning_rate):
    # TODO: Execute one DPO gradient-descent update; return updated params + metrics
    loss, grad = dpo_loss_grad(params, batch, ref_logprobs_batch, beta)

    new_params = {}
    for key in params:
        new_params[key] = params[key] - learning_rate*grad[key]

    metrics = dict(loss=loss)

    return new_params, metrics

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

