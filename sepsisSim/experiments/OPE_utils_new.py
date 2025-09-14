import numpy as np
import pandas as pd
import numpy_indexed as npi
import joblib
from tqdm import tqdm
import itertools
import copy

NSTEPS = H = 20       # max episode length in historical data
G_min = -1        # the minimum possible return
G_max =  1        # the maximum possible return
nS, nA = 1442, 8

##################
## Preparations ##
##################

def format_data_tensor(df_data, id_col='pt_id'):
    """
    Converts data from a dataframe to a tensor
    - df_data: pd.DataFrame with columns [id_col, Time, State, Action, Reward, NextState]
        - id_col specifies the index column to group episodes
    - data_tensor: integer tensor of shape (N, NSTEPS, 5) with the last last dimension being [t, s, a, r, s']
    """
    data_dict = dict(list(df_data.groupby(id_col)))
    N = len(data_dict)
    data_tensor = np.zeros((N, NSTEPS, 5), dtype=float)
    data_tensor[:, :, 2] = -1 # initialize all actions to -1
    data_tensor[:, :, 1] = -1 # initialize all states to -1
    data_tensor[:, :, 4] = -1 # initialize all next states to -1

    for i, (pt_id, df_values) in tqdm(enumerate(data_dict.items()), disable=True):
        values = df_values.set_index(id_col).values
        data_tensor[i, :len(values), :] = values
    return data_tensor

def compute_behavior_policy(df_data):
    """
    Calculate probabilities of the behavior policy π_b
    using Maximum Likelihood Estimation (MLE)
    """
    # Compute empirical behavior policy from data
    π_b = np.zeros((nS, nA))
    sa_counts = df_data.groupby(['State', 'Action']).count()[['Reward']].rename(columns={'Reward': 'count'}).reset_index()

    for i, row in sa_counts.iterrows():
        s, a = row['State'], row['Action']
        count = row['count']
        if row['Action'] == -1:
            π_b[s, :] = count
        else:
            π_b[s, a] = count

    # assume uniform action probabilities in unobserved states
    unobserved_states = (π_b.sum(axis=-1) == 0)
    π_b[unobserved_states, :] = 1

    # normalize action probabilities
    π_b = π_b / π_b.sum(axis=-1, keepdims=True)

    return π_b

def compute_behavior_policy_h(df_data):
    """
    Calculate probabilities of the behavior policy π_b
    using Maximum Likelihood Estimation (MLE)
    """
    # Compute empirical behavior policy from data
    πh_b = np.zeros((H, nS, nA))
    hsa_counts = df_data.groupby(['Time', 'State', 'Action']).count()[['Reward']].rename(columns={'Reward': 'count'}).reset_index()

    for i, row in hsa_counts.iterrows():
        h, s, a = row['Time'], row['State'], row['Action']
        count = row['count']
        if row['Action'] == -1:
            πh_b[h, s, :] = count
        else:
            πh_b[h, s, a] = count

    # assume uniform action probabilities in unobserved states
    unobserved_states = (πh_b.sum(axis=-1) == 0)
    πh_b[unobserved_states, :] = 1

    # normalize action probabilities
    πh_b = πh_b / πh_b.sum(axis=-1, keepdims=True)

    return πh_b

#########################
## Evaluating a policy ##
#########################

def policy_eval_analytic(P, R, π, γ):
    """
    Given the MDP model transition probability P (S,A,S) and reward function R (S,A),
    Compute the value function of a stochastic policy π (S,A) using matrix inversion
    
        V_π = (I - γ P_π)^-1 R_π
    """
    nS, nA = R.shape
    R_π = np.sum(R * π, axis=1)
    P_π = np.sum(P * np.expand_dims(π, 2), axis=1)
    V_π = np.linalg.inv(np.eye(nS) - γ * P_π) @ R_π
    return V_π

def policy_eval_analytic_finite(P, R, π, γ, H):
    """
    Given the MDP model transition probability P (S,A,S) and reward function R (S,A),
    Compute the value function of a stochastic policy π (S,A) using the power series formula
    Horizon h=1...H
        V_π(h) = R_π + γ P_π R_π + ... + γ^{h-1} P_π^{h-1} R_π
    """
    nS, nA = R.shape
    R_π = np.sum(R * π, axis=1)
    P_π = np.sum(P * np.expand_dims(π, 2), axis=1)
    V_π = [R_π]
    for h in range(1,H):
        V_π.append(R_π + γ * P_π @ V_π[-1])
    return list(reversed(V_π))

def OPE_IS_h(data, π_b, π_e, γ, epsilon=0.01):
    """
    - π_b, π_e: behavior/evaluation policy, shape (S,A)
    """
    # Get a soft version of the evaluation policy for WIS
    π_e_soft = np.copy(π_e).astype(float)
    π_e_soft[π_e_soft == 1] = (1 - epsilon)
    π_e_soft[π_e_soft == 0] = epsilon / (nA - 1)
    
    # Apply WIS
    return _is_h(data, π_b, π_e_soft, γ)

def _is_h(data, π_b, π_e, γ):
    """
    Weighted Importance Sampling for Off-Policy Evaluation
        - data: tensor of shape (N, T, 5) with the last last dimension being [t, s, a, r, s']
        - π_b:  behavior policy
        - π_e:  evaluation policy (aka target policy)
        - γ:    discount factor
    """
    t_list = data[..., 0].astype(int)
    s_list = data[..., 1].astype(int)
    a_list = data[..., 2].astype(int)
    r_list = data[..., 3].astype(float)
    
    # Per-trajectory returns (discounted cumulative rewards)
    G = (r_list * np.power(γ, t_list)).sum(axis=-1)
    
    # Per-transition importance ratios
    p_b = π_b[t_list, s_list, a_list]
    p_e = π_e[s_list, a_list]

    # Deal with variable length sequences by setting ratio to 1
    terminated_idx = (a_list == -1)
    p_b[terminated_idx] = 1
    p_e[terminated_idx] = 1
    
    if not np.all(p_b > 0):
        import pdb
        pdb.set_trace()
    assert np.all(p_b > 0), "Some actions had zero prob under p_b, WIS fails"

    # Per-trajectory cumulative importance ratios, take the product
    rho = (p_e / p_b).prod(axis=1)
    rho_norm = rho / rho.sum()

    # directly calculate weighted average over trajectories
    is_value = np.average(G*rho) # (G @ rho) / len(G)
    wis_value = np.average(G, weights=rho) # (G @ rho_norm)
    ess1 = 1 / (rho_norm ** 2).sum()
    ess1_ = (rho.sum()) ** 2 / ((rho ** 2).sum())
    assert np.isclose(ess1, ess1_)
    ess2 = 1. / rho_norm.max()
    return is_value, wis_value, {
        'ESS1': ess1, 'ESS2': ess2, 'G': G,
        'rho': rho, 'rho_norm': rho_norm
    }


## DR:
def compute_empirical_q_h(df_data, pi_e, gamma, H):
    """
    Compute empirical Q_h and V_h using backward pass fitted Q-evaluation (non-parametric).
    - df_data: pd.DataFrame with columns ['Time', 'State', 'Action', 'Reward', 'NextState']
    - pi_e: evaluation policy, shape (nS, nA)
    - Returns: Q_hats list[H] of (nS, nA), V_hats list[H] of (nS,)
    """
    Q_hats = [np.zeros((nS, nA)) for _ in range(H)]
    V_hats = [np.zeros(nS) for _ in range(H)]
    V_next = np.zeros(nS)  # V_H = 0

    for h in range(H - 1, -1, -1):
        df_h = df_data[df_data['Time'] == h].copy()
        if len(df_h) == 0:
            V_hats[h] = np.zeros(nS)
            V_next = V_hats[h].copy()
            continue

        # Vectorize v_next with mask to avoid invalid indexing
        mask_invalid = (df_h['NextState'] < 0) | (df_h['NextState'] >= nS)
        df_h['v_next'] = 0.0
        valid_mask = ~mask_invalid
        if valid_mask.any():
            valid_next = df_h.loc[valid_mask, 'NextState'].values
            df_h.loc[valid_mask, 'v_next'] = V_next[valid_next] * gamma

        # Group by State, Action (only for a >= 0)
        df_h_valid = df_h[df_h['Action'] >= 0]
        if len(df_h_valid) == 0:
            V_hats[h] = np.zeros(nS)
            V_next = V_hats[h].copy()
            continue

        grouped = df_h_valid.groupby(['State', 'Action'])
        for (s, a), group in grouped:
            if len(group) == 0:
                continue
            mean_r = group['Reward'].mean()
            mean_v_next = group['v_next'].mean()
            Q_hats[h][s, a] = mean_r + mean_v_next

        # Compute V_h(s) = sum_a pi_e(s, a) * Q_h(s, a)
        for s in range(nS):
            V_hats[h][s] = np.sum(pi_e[s] * Q_hats[h][s])

        V_next = V_hats[h].copy()

    return Q_hats, V_hats

def compute_empirical_q_h_annot(df_annot, pi_e, gamma, H, version='v1'):
    """
    Compute empirical Q_h and V_h using annotated data for CANDOR DM+-IS.
    - df_annot: pd.DataFrame with columns ['Time', 'State', 'Action', 'Reward', 'NextState']
        Augmented with counterfactual transitions (Action = a_alt, Reward adjusted to Q(s,a_alt))
    - version: 'v1' for initial flip, 'v2' for all steps
    - Fits Q using all transitions (original + annotated)
    """
    Q_hats = [np.zeros((nS, nA)) for _ in range(H)]
    V_hats = [np.zeros(nS) for _ in range(H)]
    V_next = np.zeros(nS)  # V_H = 0

    for h in range(H - 1, -1, -1):
        # Filter transitions at time h (valid actions)
        df_h = df_annot[(df_annot['Time'] == h) & (df_annot['Action'] >= 0)].copy()
        if len(df_h) == 0:
            V_hats[h] = np.zeros(nS)
            V_next = V_hats[h].copy()
            continue

        # Vectorize v_next with mask to avoid invalid indexing
        mask_invalid = (df_h['NextState'] < 0) | (df_h['NextState'] >= nS)
        df_h['v_next'] = 0.0
        valid_mask = ~mask_invalid
        if valid_mask.any():
            valid_next = df_h.loc[valid_mask, 'NextState'].values
            df_h.loc[valid_mask, 'v_next'] = V_next[valid_next] * gamma

        # Group by State, Action
        grouped = df_h.groupby(['State', 'Action'])
        counts = {}  # To average properly
        for (s, a), group in grouped:
            if len(group) == 0:
                continue
            mean_r = group['Reward'].mean()
            mean_v_next = group['v_next'].mean()
            Q_hats[h][s, a] = mean_r + mean_v_next
            counts[(s, a)] = len(group)

        # Compute V_h(s) = sum_a pi_e(s, a) * Q_h(s, a)
        for s in range(nS):
            V_hats[h][s] = np.sum(pi_e[s] * Q_hats[h][s])

        V_next = V_hats[h].copy()

    return Q_hats, V_hats

def _dr_h(data, pi_b, pi_e, Q_hats, V_hats, gamma):
    """
    Doubly Robust for Off-Policy Evaluation (trajectory-level)
    - data: tensor (N, T, 5) [t, s, a, r, s']
    - pi_b: (H, nS, nA)
    - pi_e: (nS, nA)
    - Q_hats, V_hats: lists of length H
    - Returns: dr_value, info dict with 'dr_returns'
    """
    t_list = data[..., 0].astype(int)
    s_list = data[..., 1].astype(int)
    a_list = data[..., 2].astype(int)
    r_list = data[..., 3].astype(float)
    sp_list = data[..., 4].astype(int)

    N, T = data.shape[:2]
    dr_returns = np.zeros(N)

    for i in range(N):
        traj_dr = 0.0
        cum_rho = 1.0
        for tt in range(T):
            t = int(t_list[i, tt])
            if t >= H:
                break
            s = int(s_list[i, tt])
            a = int(a_list[i, tt])
            r = r_list[i, tt]
            sp = int(sp_list[i, tt])

            if a == -1:
                rho_t = 1.0
                q_sa = 0.0
                v_sp = 0.0
            else:
                p_b_t = pi_b[t, s, a]
                if p_b_t == 0:
                    rho_t = 0.0  # or handle, but assume >0
                else:
                    rho_t = pi_e[s, a] / p_b_t
                q_sa = Q_hats[t][s, a]
                v_sp = 0.0 if (sp < 0 or sp >= nS) else (V_hats[t + 1][sp] if t + 1 < H else 0.0)

            delta = r + gamma * v_sp - q_sa
            v_s = V_hats[t][s]
            contrib = v_s + cum_rho * rho_t * delta
            traj_dr += (gamma ** t) * contrib
            cum_rho *= rho_t

        dr_returns[i] = traj_dr

    dr_value = np.mean(dr_returns)
    return dr_value, {
        'dr_returns': dr_returns
    }

def _candor_dm_is_h(data, pi_b, pi_e, Q_hats, V_hats, gamma):
    """
    CANDOR DM+-IS: Doubly Robust with augmented DM from annotations (trajectory-level)
    - Similar to _dr_h, but Q_hats/V_hats from annotated data
    """
    t_list = data[..., 0].astype(int)
    s_list = data[..., 1].astype(int)
    a_list = data[..., 2].astype(int)
    r_list = data[..., 3].astype(float)
    sp_list = data[..., 4].astype(int)

    N, T = data.shape[:2]
    candor_returns = np.zeros(N)

    for i in range(N):
        traj_candor = 0.0
        cum_rho = 1.0
        for tt in range(T):
            t = int(t_list[i, tt])
            if t >= H:
                break
            s = int(s_list[i, tt])
            a = int(a_list[i, tt])
            r = r_list[i, tt]
            sp = int(sp_list[i, tt])

            if a == -1:
                rho_t = 1.0
                q_sa = 0.0
                v_sp = 0.0
            else:
                p_b_t = pi_b[t, s, a]
                if p_b_t == 0:
                    rho_t = 0.0
                else:
                    rho_t = pi_e[s, a] / p_b_t
                q_sa = Q_hats[t][s, a]
                v_sp = 0.0 if (sp < 0 or sp >= nS) else (V_hats[t + 1][sp] if t + 1 < H else 0.0)

            delta = r + gamma * v_sp - q_sa
            v_s = V_hats[t][s]
            contrib = v_s + cum_rho * rho_t * delta
            traj_candor += (gamma ** t) * contrib
            cum_rho *= rho_t

        candor_returns[i] = traj_candor

    candor_value = np.mean(candor_returns)
    return candor_value, {
        'candor_returns': candor_returns
    }

def OPE_DR_h(data, pi_b, pi_e, Q_hats, V_hats, gamma, epsilon=0.0):
    """
    Doubly Robust Off-Policy Evaluation
    - Similar to OPE_IS_h, but uses Q_hats and V_hats
    - epsilon: not used for DR, kept for consistency
    """
    return _dr_h(data, pi_b, pi_e, Q_hats, V_hats, gamma)

def OPE_CANDOR_h(data, pi_b, pi_e, Q_hats, V_hats, gamma, epsilon=0.0, version='v1'):
    """
    CANDOR DM+-IS Off-Policy Evaluation
    - Uses annotated Q_hats/V_hats in DR framework
    - version: passed to compute_empirical_q_h_annot if needed, but here assumed precomputed
    """
    return _candor_dm_is_h(data, pi_b, pi_e, Q_hats, V_hats, gamma)