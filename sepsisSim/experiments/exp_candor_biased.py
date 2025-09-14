# ## Simulation parameters
exp_name = 'exp-CANDOR-22'
eps = 0.10
eps_str = '0_1'

run_idx_length = 1_000
N_val = 1_000
runs = 50

# Number of action-flipped states
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--flip_num', type=int)
parser.add_argument('--flip_seed', type=int)
parser.add_argument('--noise', type=float, default=0.2)
args = parser.parse_args()
pol_flip_num = args.flip_num
pol_flip_seed = args.flip_seed
annot_noise = args.noise

pol_name = f'flip{pol_flip_num}_seed{pol_flip_seed}'
out_fname = f'./results/{exp_name}/vaso_eps_{eps_str}-{pol_name}-candor-biased-Noise_{annot_noise}.csv'

import numpy as np
import pandas as pd

df_tmp = None
try:
    df_tmp = pd.read_csv(out_fname)
except:
    pass

if df_tmp is not None:
    print('File exists')
    quit()

from tqdm import tqdm
from collections import defaultdict
import pickle
import itertools
import copy
import random
import itertools
import joblib
from joblib import Parallel, delayed

from OPE_utils_new import (
    format_data_tensor,
    policy_eval_analytic_finite,
    OPE_CANDOR_h,
    compute_behavior_policy_h,
    compute_empirical_q_h_annot,
)


def policy_eval_helper(π):
    V_H = policy_eval_analytic_finite(P.transpose((1,0,2)), R, π, gamma, H)
    Q_H = [(R + gamma * P.transpose((1,0,2)) @ V_H[h]) for h in range(1,H)] + [R]
    J = isd @ V_H[0]
    # Check recursive relationships
    assert len(Q_H) == H
    assert len(V_H) == H
    assert np.all(Q_H[-1] == R)
    assert np.all(np.sum(π * Q_H[-1], axis=1) == V_H[-1])
    assert np.all(R + gamma * P.transpose((1,0,2)) @ V_H[-1] == Q_H[-2])
    return V_H, Q_H, J

def iqm(x):
    return scipy.stats.trim_mean(x, proportiontocut=0.25, axis=None)

NSTEPS = H = 20   # max episode length in historical data # Horizon of the MDP
G_min = -1        # the minimum possible return
G_max =  1        # the maximum possible return
nS, nA = 1442, 8

PROB_DIAB = 0.2

# Ground truth MDP model
MDP_parameters = joblib.load('../data/MDP_parameters.joblib')
P = MDP_parameters['transition_matrix_absorbing'] # (A, S, S_next)
R = MDP_parameters['reward_matrix_absorbing_SA'] # (S, A)
nS, nA = R.shape
gamma = 0.99

# unif rand isd, mixture of diabetic state
isd = joblib.load('../data/modified_prior_initial_state_absorbing.joblib')
isd = (isd > 0).astype(float)
isd[:720] = isd[:720] / isd[:720].sum() * (1-PROB_DIAB)
isd[720:] = isd[720:] / isd[720:].sum() * (PROB_DIAB)

# Precomputed optimal policy
π_star = joblib.load('../data/π_star.joblib')



# ## Load data
input_dir = f'../datagen/vaso_eps_{eps_str}-100k/'

def load_data(fname):
    print('Loading data', fname, '...', end='')
    df_data = pd.read_csv('{}/{}'.format(input_dir, fname)).rename(columns={'State_idx': 'State'})#[['pt_id', 'Time', 'State', 'Action', 'Reward']]

    # Assign next state
    df_data['NextState'] = [*df_data['State'].iloc[1:].values, -1]
    df_data.loc[df_data.groupby('pt_id')['Time'].idxmax(), 'NextState'] = -1
    df_data.loc[(df_data['Reward'] == -1), 'NextState'] = 1440
    df_data.loc[(df_data['Reward'] == 1), 'NextState'] = 1441

    assert ((df_data['Reward'] != 0) == (df_data['Action'] == -1)).all()

    print('DONE')
    return df_data


# df_train = load_data('1-features.csv') # tr
df_seed2 = load_data('2-features.csv') # va


# ## Load annotations (perfect CF, to be noised)
df_annot_all = pd.read_pickle(f'results/vaso_eps_{eps_str}-evalOpt_df_seed2_aug_step.pkl')
print('Loaded perfect annotations:', len(df_annot_all), 'rows')


## Default weighting scheme for CANDOR (same as original for consistency)

weight_a_sa = np.zeros((nS, nA, nA))

# default weight if no counterfactual actions
for a in range(nA):
    weight_a_sa[:, a, a] = 1

# split equally between factual and counterfactual actions
for s in range(nS):
    a = π_star.argmax(axis=1)[s]
    a_tilde = a+1-2*(a%2)
    weight_a_sa[s, a, a] = 0.5
    weight_a_sa[s, a, a_tilde] = 0.5
    weight_a_sa[s, a_tilde, a] = 0.5
    weight_a_sa[s, a_tilde, a_tilde] = 0.5

assert np.all(weight_a_sa.sum(axis=-1) == 1)


# ## Policies

# vaso unif, mv abx optimal
π_unif = (np.tile(π_star.reshape((-1,2,2,2)).sum(axis=3, keepdims=True), (1,1,1,2)).reshape((-1, 8)) / 2)


# ### Behavior policy

# vaso eps=0.5, mv abx optimal
π_beh = (np.tile(π_star.reshape((-1,2,2,2)).sum(axis=3, keepdims=True), (1,1,1,2)).reshape((-1, 8)) / 2)
π_beh[π_star == 1] = 1-eps
π_beh[π_beh == 0.5] = eps

V_H_beh, Q_H_beh, J_beh = policy_eval_helper(π_beh)
J_beh


# ### Optimal policy
V_H_star, Q_H_star, J_star = policy_eval_helper(π_star)
J_star


# ### flip action for x% states

rng_flip = np.random.default_rng(pol_flip_seed)
flip_states = rng_flip.choice(range(1440), pol_flip_num, replace=False)

π_tmp = (np.tile(π_star.reshape((-1,2,2,2)).sum(axis=3, keepdims=True), (1,1,1,2)).reshape((-1, 8)) / 2)
π_flip = π_tmp.copy()
π_flip[π_tmp == 0.5] = 0
π_flip[π_star == 1] = 1
for s in flip_states:
    π_flip[s, π_tmp[s] == 0.5] = 1
    π_flip[s, π_star[s] == 1] = 0
assert π_flip.sum(axis=1).mean() == 1

np.savetxt(f'./results/{exp_name}/policy_{pol_name}.txt', π_flip)


# ## Compare OPE

π_eval = π_flip


# ### Proposed: replace future with the value function for the evaluation policy
rng_annot = np.random.default_rng(seed=123456789)

V_H_eval, Q_H_eval, J_eval = policy_eval_helper(π_eval)

# Create biased annotations for Q fitting (vectorized)
df_annot = df_annot_all.copy()
df_annot['Weight'] = np.nan

# Precompute a_f for CF rows (vectorized flip)
mask_cf = df_annot['NextState'] == 1442
if mask_cf.any():
    a_cf = df_annot.loc[mask_cf, 'Action'].astype(int).values
    s_cf = df_annot.loc[mask_cf, 'State'].astype(int).values
    h_cf = df_annot.loc[mask_cf, 'Time'].astype(int).values

    # Vectorized a_f = a_cf + 1 - 2*(a_cf % 2)
    a_f = a_cf + 1 - 2 * (a_cf % 2)

    # Vectorized weights using precomputed weight_a_sa (broadcast)
    weights_cf = weight_a_sa[s_cf, a_f, a_cf]  # Shape: (num_cf,)
    df_annot.loc[mask_cf, 'Weight'] = weights_cf

    # Vectorized noise addition to Reward (handle list indexing for Q_H_eval)
    noise = rng_annot.normal(0, annot_noise, size=len(a_cf))
    q_values = np.array([Q_H_eval[h][s, a] for h, s, a in zip(h_cf, s_cf, a_cf)])
    df_annot.loc[mask_cf, 'Reward'] = q_values + noise

# Terminating rows
mask_term = df_annot['NextState'].isin([1440, 1441])
df_annot.loc[mask_term, 'Weight'] = 1.0

# Fillna
df_annot['Weight'] = df_annot['Weight'].fillna(1)


# ## Train Q-model on biased annotated data
print('Computing empirical Q_h and V_h from biased annotations...')
Q_hats, V_hats = compute_empirical_q_h_annot(df_annot, π_eval, gamma, H, version='v1')


# ### CANDOR DM+-IS (biased): on val dataset (original trajectories)

df_results = []
for run in range(runs):
    df_va = df_seed2.set_index('pt_id').loc[200000+run*run_idx_length:200000+run*run_idx_length + N_val - 1].reset_index()[
        ['pt_id', 'Time', 'State', 'Action', 'Reward', 'NextState']
    ]
    df = df_va[['pt_id', 'Time', 'State', 'Action', 'Reward', 'NextState']]

    # OPE - CANDOR prep
    data_va = format_data_tensor(df)
    pi_b_va = compute_behavior_policy_h(df)

    # OPE - CANDOR (biased)
    CANDOR_value, info = OPE_CANDOR_h(data_va, pi_b_va, π_eval, Q_hats, V_hats, gamma, epsilon=0.0, version='v1')
    df_results.append([CANDOR_value])

df_results = pd.DataFrame(df_results, columns=['CANDOR_value'])
df_results.to_csv(out_fname, index=False)