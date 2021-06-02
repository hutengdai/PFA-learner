import sys
import csv
import random
import itertools
import functools
import argparse

import numpy as np
import scipy.special
import pandas as pd
import torch
import einops

try:
    import genbmm
    import entmax
except ModuleNotFoundError:
    genbmm = None
    entmax = None

LOG2 = np.log(2)
BOUNDARY_SYMBOL_INDEX = 0
INITIAL_STATE_INDEX = 0
INF = float('inf')
EOS = '!!</S>!!'

EPSILON = 10 ** -8
DEFAULT_NUM_EPOCHS = 10 ** 5
DEFAULT_NUM_STATES = 500
DEFAULT_BATCH_SIZE = 5
DEFAULT_PRINT_EVERY = 1000
DEFAULT_NUM_SAMPLES = 0
DEFAULT_DATA_SEED = 0
DEFAULT_INIT_TEMPERATURE = 1
DEFAULT_PERM_TEST_NUM_SAMPLES = 0
DEFAULT_ACTIVATION = "softmax" # sparsemax and entmax15 are available but numerically unstable
DEFAULT_LR = 0.001

newaxis = None
colon = slice(None)

def prod(xs):
    y = 1
    for x in xs:
        y *= x
    return y

def buildup(iterable):
    """ Build up

    Example:
    >>> list(buildup("abcd"))
    [('a',), ('a', 'b'), ('a', 'b', 'c'), ('a', 'b', 'c', 'd')]

    """
    so_far = []
    for x in iterable:
        so_far.append(x)
        yield tuple(so_far)

def memoize(f):
    cache = {}

    def wrapper(*args):
        if args in cache:
            return cache[args]
        else:
            cache[args] = result = f(*args)
            return result

    wrapper.cache = cache
    wrapper = functools.wraps(f)(wrapper)
    
    return wrapper        

def stationary(M):
    """ Find the stationary distribution in a way that is differentiable """
    n = M.shape[-1]
    k = M.ndim
    A = M.transpose(-1, -2) - torch.eye(n, device=M.device)
    colons = (colon,)*(k-2)
    A[colons+(-1, colon,)] = 1 # A[:, -1, :]
    against = torch.zeros(n, device=M.device)
    against[-1] = 1
    return torch.solve(against[:, newaxis], A).solution.T[-1].T  # craziness

def identity(x):
    return x

class PFA:
    # Define the p-space field:
    zero = 0
    one = 1
    mul = torch.mul
    matmul = torch.matmul
    sum = torch.sum
    prod = torch.prod
    div = torch.div
    to_log = torch.log

    def to_exp(self, x):
        return x

    @property
    def inventory(self):
        return range(self.num_symbols)

    def entropy(self, ps, eps=EPSILON):
        logs = self.to_log(ps+eps)
        return -(ps * logs).sum() / LOG2    

    def mutual_information(self, p_xy):
        p_y = self.sum(p_xy, axis=-2)
        p_x = self.sum(p_xy, axis=-1)
        return self.entropy(p_y) + self.entropy(p_x) - self.entropy(p_xy)

    def one_hot(self, k, n):
        y = torch.full((n,), self.zero, dtype=self.E[0].dtype, device=self.device)
        y[k] = self.one
        return y

    def conditional_logp(self, context, symbol, **kwds):
        logp_context = self.logp_symbol_sequence(context, **kwds)
        continuation = tuple(context) + (symbol,)
        logp_full = self.logp_symbol_sequence(continuation, **kwds)
        return logp_full - logp_context

    @property
    def memory_mi(self):
        """ I[Q' : Q, X] """
        _, I_QX, I_QQ, synergy = self.state_information_lattice
        return I_QX + I_QQ + synergy

    @property
    def nondeterminism(self):
        """ H[Q' | Q, X] """
        H_Q_QX, *_ = self.state_information_lattice
        return H_Q_QX

    def __add__(self, other):
        return self.product(other)


    # Things below need to be implemented for a concrete class

    def product(self, other):
        raise NotImplementedError

    def generate(self, *a, **k):
        raise NotImplementedError

    def logp_symbol_sequence(self, *a, **k):
        raise NotImplementedError

    @property
    def state_information_lattice(self):
        return torch.full((4,), float('nan'))

    @property
    def num_states(self):
        raise NotImplementedError

    @property
    def num_factors(self):
        raise NotImplementedError

    @property
    def num_symbols(self):
        raise NotImplementedError

    
class SimplePFA(PFA):
    def __init__(self, E, T, starting_state=None, starting_state_distro=None, device=None):
        """ A PFA over an alphabet of symbols X with states Q. 

        T : a transition matrix of shape Q x X x Q, where T[i,j,k] = p(q_k | q_j, x_k)
        E : an emission matrix of shape Q x X, where E[i,j] = p(x_j | q_i)

        """
        *_, Q1T, XT, Q2T = T.shape
        assert Q1T == Q2T, "T must have shape Q x X x Q"
        *_, QE, XE = E.shape
        assert QE == Q1T, "E and T must have the same number of states"
        assert XE == XT, "E and T must have the same number of symbols"

        assert isinstance(E, torch.Tensor)

        self.T = T
        self.E = E

        self.device = device

        assert starting_state_distro is None or starting_state is None
        if starting_state_distro is None and starting_state is None:
            starting_state_distro = self.one_hot(INITIAL_STATE_INDEX, self.num_states)
        elif starting_state is not None:
            # if an individual starting state is provided, then start from a one-hot distribution on that state
            starting_state_distro = self.one_hot(starting_state, self.num_states)
        self.starting_state_distro = starting_state_distro

    def logp_symbol_sequence(self, sequence, starting_state_distro=None, starting_state=None):
        """ Log probability of sequence of symbols.""" 
        assert starting_state_distro is None or starting_state is None
        if starting_state_distro is None and starting_state is None:
            starting_state_distro = self.starting_state_distro
        elif starting_state is not None:
            # if an individual starting state is provided, then start from a one-hot distribution on that state
            starting_state_distro = self.starting_state_distro
        # Use the forward algorithm.        
        sequence = tuple(sequence)
        q = starting_state_distro
        logp = 0.0
        for x in sequence:
            logp += self.to_log(self.matmul(q, self.E)[x])
            q = self.matmul(q, self.T[:, x, :])
        return logp

    @property
    def num_factors(self):
        return 1

    @property
    def num_states(self):
        return self.E.shape[-2]

    @property
    def num_symbols(self):
        return self.E.shape[-1]

    def state_matrix(self, sequence, starting_state_distro=None):
        """ 
        Input: A sequence of symbols X^T and a distribution on starting states (vector of size Q).
        Output: A matrix T x Q where Q[t,q] = p(q_t | x_{<t}).

        Optionally, you can also provide distribution over starting states (vector of length Q).
        """
        if starting_state_distro is None:
            q_t = self.stationary_Q
        else:
            q_t = starting_state_distro
        rows = [q_t]
        for x_t in sequence:
            q_t = self.matmul(q_t, self.T[:, x_t, :])
            rows.append(q_t)
        return torch.stack(rows)

    @property
    def state_transition_matrix(self):
        """ a Q x Q matrix giving p(q_{t+1} | q_t) """
        # T is shape Q x X x Q'
        # E is shape Q x X
        ET = self.mul(self.E.unsqueeze(-1), self.T)
        return self.sum(ET, axis=-2)

    @property
    def stationary_Q(self):
        """ Vector of length q giving the stationary p(q) """
        return stationary(self.to_exp(self.state_transition_matrix))

    @property
    def stationary_X(self):
        """ A vector giving p(x) """
        return self.matmul(self.stationary_Q, self.E)

    @property
    def state_information_lattice(self):
        """ H[Q_{t+1} | Q_t, X_t], I[Q_{t+1}:X_t], I[Q_{t+1}:Q_t], and -I[Q_{t+1}:Q_t:X_t] 
        These sum to H[Q].
        """
        H_Q = self.entropy(self.stationary_Q)
        
        # Next I[Q_{t+1} : X_t]
        p_QX = self.mul(self.stationary_Q.unsqueeze(-1), self.E)
        I_QX = self.mutual_information(p_QX)

        
        p_QXQ = self.mul(p_QX.unsqueeze(-1), self.T)
        p_QQ = self.sum(p_QXQ, axis=-2) 
        I_QQ = self.mutual_information(p_QQ) 
        
        H_Q_QX = self.entropy(p_QXQ) - self.entropy(p_QX)
        
        synergy = H_Q - H_Q_QX - I_QX - I_QQ

        return torch.stack([H_Q_QX, I_QX, I_QQ, synergy], dim=-1)

    def generate(self, starting_state=INITIAL_STATE_INDEX):
        states = range(self.num_states)

        E = self.to_exp(self.E.detach().numpy())
        T = self.to_exp(self.T.detach().numpy())

        q = starting_state
        while True:
            x = np.random.choice(self.inventory, p=E[q, :])
            yield (q, x)
            if x == BOUNDARY_SYMBOL_INDEX:
                break
            q = np.random.choice(states, p=T[q, x, :])

class PDFA(PFA):
    @property
    def starting_state(self):
        assert self.entropy(self.starting_state_distro) == 0
        return self.starting_state_distro.argmax()
    
    @property
    def nondeterminism(self):
        return torch.zeros(1)

class SimplePDFA(SimplePFA, PDFA):
    @property
    def entropy_rate(self):
        H_Q = self.entropy(self.stationary_Q)
        QE = self.mul(self.stationary_Q.unsqueeze(-1), self.E)
        H_QE = self.entropy(QE)
        return H_QE - H_Q

    def factored(self):
        return FactorPDFA([self.E], [self.T], [self.starting_state], device=self.device)

    def product(self, other):
        factorized_self = self.factored()
        return factorized_self + other

    @property
    def activity(self):
        return pdfa_norm(self.E)

def pdfa_norm(E):
    """ Norm of a conditional distribution according to Wen & Ray (2012: 1136) """
    minmax = E.max(-1).values.log() - E.min(-1).values.log()
    return minmax.max(-1).values    

class FactorPDFA(PDFA):
    def __init__(self, Es, Ts, starting_states=None, device=None):
        self.E = Es # list of QX
        self.T = Ts # list of QXQ
        if starting_states is None:
            self.starting_states = [INITIAL_STATE_INDEX for _ in range(self.num_factors)]
        else:
            self.starting_states = starting_states
        self.device = device

    @property
    def num_factors(self):
        return len(self.E)

    @property
    def num_states(self):
        return prod(sub_T.shape[-1] for sub_T in self.T)

    @property
    def num_symbols(self):
        return self.E[0].shape[-1]

    @property
    def factor_norms(self):
        """ Norm of a PDFA according to Wen & Ray (2012: 1136) """
        return list(map(pdfa_norm, self.E))

    def product(self, other):
        """ Return product with another factor machine """
        if isinstance(other, SimplePDFA):
            other = other.factored()
        starting_states = list(self.starting_states) + list(other.starting_states)
        E = self.E + list(other.E)
        T = self.T + list(other.T)
        return FactorPDFA(E, T, starting_states=starting_states, device=self.device)

    def logp_symbol_sequence(self, sequence, starting_states=None):
        if starting_states is None:
            starting_states = self.starting_states
        num_factors_states = [sub_T.shape[-1] for sub_T in self.T]
        qs = [
            self.one_hot(starting_state, num_factor_states)
            for starting_state, num_factor_states in zip(starting_states, num_factors_states)
        ]
        # states are like [ [1, 0],  [1, 0], ..., [0, 1], ...] for each sub-automaton
        # we want geometric mixture p(x|states) = 1/Z \prod_a p_a(x | states[a])
        logp = 0.0
        for segment in sequence:
            sub_probs = torch.stack([self.matmul(q, sub_E) for q, sub_E in zip(qs, self.E)]) # shape AX
            log_numerators = self.to_log(sub_probs).sum(-2) # shape X
            logZ = log_numerators.logsumexp(-1) # scalar
            logp += log_numerators[segment] - logZ
            qs = [self.matmul(q, sub_T[:, segment, :]) for q, sub_T in zip(qs, self.T)]
        return logp

class HomogenousFactorPDFA(FactorPDFA):
    """ Factor machine where each factor has the same number of states.
    This enables much faster computations. """
    
    @property
    def num_factors(self):
        return self.E.shape[-3]

    @property
    def num_states(self):
        return self.E.shape[-2] ** self.num_factors

    @property
    def num_symbols(self):
        return self.E.shape[-1]

    @property
    def factor_norms(self):
        """ Norm of a PDFA according to Wen & Ray (2012: 1136) """
        return pdfa_norm(self.E)

    def logp_symbol_sequence(self, sequence, starting_states=None):
        if starting_states is None:
            starting_states = self.starting_states

        num_factor_states = self.T.shape[-1]
        
        qs = torch.stack([
            self.one_hot(starting_state, num_factor_states)
            for starting_state in starting_states
        ])
        # states are like [ [1, 0],  [1, 0], ..., [0, 1], ...] for each sub-automaton
        # we want geometric mixture p(x|states) = 1/Z \prod_a p_a(x | states[a])
        qs = qs.unsqueeze(-2) # shape A1Q -- need this shape for stuff below to work
        logp = 0.0
        for segment in sequence:
            sub_probs = self.matmul(qs, self.E).squeeze(-2) # shape AX
            log_numerators = self.to_log(sub_probs).sum(-2) # shape X
            logZ = log_numerators.logsumexp(-1) # scalar
            logp += log_numerators[segment] - logZ
            qs = self.matmul(qs, self.T[:, :, segment, :])
        return logp
    
    def product(self, other):
        """ Return product with another factor machine with the same number of states in each sub-automaton """
        if isinstance(other, HomogenousFactorPDFA):
            starting_states = list(self.starting_states) + list(other.starting_states)
            E = torch.cat([self.E, other.E], dim=-3)
            T = torch.cat([self.T, other.T], dim=-4)
            return HomogenousFactorPDFA(E, T, starting_states=starting_states, device=self.device)
        else:
            return other + self # use the class of other
    
    # CONSTRUCTORS

    @classmethod
    def featural_pdfa(cls, T_generator, Es, k, device=None):
        num_factors = Es.shape[0]
        num_symbols = Es.shape[-1]

        starting_states = [0 for _ in range(num_factors)]
        starting_states[BOUNDARY_SYMBOL_INDEX] = 1

        # start with an inactive E matrix
        E = torch.full((num_factors, k, num_symbols), cls.div(1.0, num_symbols)) # or k+1 for LT/PT?
        # parameters populate the E matrix only for the active state: 
        E[:, -1, :] = Es # shape AQX
        
        T = T_generator(k, num_factors) # or k+1 for LT/PT?
        
        assert T.shape[-1] == T.shape[-3] == 2        
        assert T.shape[-2] == num_symbols
        assert T.shape[-4] == num_factors
        
        return cls(E.to(device), T.to(device), starting_states=starting_states, device=device)

    @classmethod
    def sp_sl(cls, Es, sp_k=2, sl_k=2, device=None):
        SP_Es, SL_Es = Es
        sp = cls.sp(SP_Es, k=sp_k, device=device)
        sl = cls.sl(SL_Es, k=sl_k, device=device)
        return sp + sl
        
    @classmethod
    def sp(cls, Es, k=2, device=None):
        return cls.featural_pdfa(strictly_piecewise_transition_matrices, Es, k, device=device)

    @classmethod
    def sl(cls, Es, k=2, device=None):
        return cls.featural_pdfa(strictly_local_transition_matrices, Es, k, device=device)

turn_on = torch.Tensor([
    [0, 1],
    [0, 1],
])
turn_off = torch.Tensor([
    [1, 0],
    [1, 0],
])
stay = torch.Tensor([
    [1, 0],
    [0, 1],
])
flip = torch.Tensor([
    [0, 1],
    [1, 0],
])

@memoize
def old_strictly_piecewise_transition_matrices(k, n):
    """ Generate transition matrices for 2-SP factor machines over inventory of n symbols """
    T = torch.stack([stay for _ in range(n)]) # shape AQQ'
    T = T.unsqueeze(-2).repeat(1, 1, n, 1) # shape AQXQ'
    for i in range(n):
        T[i, :, i, :] = turn_on
    return T

@memoize
def old_strictly_local_transition_matrices(k, n):
    """ Generate transition matrices for k-SL factor machines over inventory of n symbols """ 
    # in state q_{ab...c}, given d, go into state q_{b...cd}
    # there will be n^{k-1} factors
    T = torch.stack([turn_off for _ in range(n)])
    T = T.unsqueeze(-2).repeat(1, 1, n, 1)
    for i in range(n):
        T[i, :, i, :] = turn_on
    return T

def one_hot(k, n):
    y = torch.zeros(n)
    y[k] = 1
    return y

def reverse_buildup(xs):
    # abcd -> iterator of tuples of [d, cd, bcd, abcd]
    return map(tuple, map(reversed, buildup(reversed(xs))))

def all_same(xs):
    x, *rest = xs
    return all(x == y for y in rest)

def strictly_local_transition_matrices(k, n):
    shape = (n,) * k + (k,k)
    num_dims = len(shape)
    T = torch.zeros(shape)
    
    goto_initial = one_hot(0, k).repeat(k, 1)
    # By default, factors turn off 
    T[num_dims*(colon,)] = goto_initial # by default always return to the initial state

    for segments in string_power(n, k-1):
        # for example, segments = abc
        for prefix in reverse_buildup(segments): 
            # for example, prefix = "bc"
            *_, current = prefix # for example, current = c
            k_prefix = len(prefix) # in the example, k_prefix = |bc| = 2
            expanded_prefix = prefix + (colon,)*(len(segments) - k_prefix) # b,c,:
            # upon seeing c, for all factors matching b,c,:, if they are in ANY state corresponding to having seen b (1),
            # then they go into the state for having seen bc (2), otherwise they do the default.
            compatible_states = [k_prefix - 1]
            if k_prefix == k - 1 and all_same(segments): # hack. what is the elegant solution?
                compatible_states.append(k_prefix)
            for state in compatible_states:
                T[expanded_prefix + (current, state)] = one_hot(k_prefix, k) 
    return einops.rearrange(T.reshape(n**(k-1), n, k, k), "f x q1 q2 -> f q1 x q2")

@memoize
def strictly_piecewise_transition_matrices(k, n):
    shape = (n,) * k + (k,k)
    num_dims = len(shape)
    T = torch.zeros(shape)
    
    stay = torch.eye(k)
    # By default, factors turn off 
    T[num_dims*(colon,)] = stay # by default always return to the initial state

    for segments in string_power(n, k-1):
        # for example, segments = abc
        for prefix in reverse_buildup(segments): 
            # for example, prefix = "bc"
            *_, current = prefix # for example, current = c
            k_prefix = len(prefix) # in the example, k_prefix = |bc| = 2
            expanded_prefix = prefix + (colon,)*(len(segments) - k_prefix) # b,c,:
            # upon seeing c, for all factors matching b,c,:, if they are in ANY state corresponding to having seen b (1),
            # then they go into the state for having seen bc (2), otherwise they do the default.
            compatible_states = [k_prefix - 1]
            if k_prefix == k - 1 and all_same(segments): # weird special case for final state ... TODO is this right for SP?
                compatible_states.append(k_prefix)
            for state in compatible_states:
                T[expanded_prefix + (current, state)] = one_hot(k_prefix, k)
    return einops.rearrange(T.reshape(n**(k-1), n, k, k), "f x q1 q2 -> f q1 x q2")

def sp_sl_slow(sp_Es, sl_E, sp_k=2, sl_k=2, device=None):
    # HomogenousFactorPDFA.sp_sl is much faster
    num_symbols = sl_E.shape[-1]
    assert num_symbols == sp_Es.shape[-1]
    assert sp_Es.shape[-2] == num_symbols ** (sp_k - 1)
    assert sl_E.shape[-2] == num_symbols ** (sl_k - 1)
    sp = HomogenousFactorPDFA.sp(sp_Es, k=sp_k, device=device)
    sl_T = strictly_local_transition_matrix(num_symbols, k=sl_k)
    sl = SimplePDFA(sl_E, sl_T, device=device)
    return sp + sl

class FastStrictlyPiecewisePDFA(HomogenousFactorPDFA):
    # 2x faster than HomogenousFactorPDFA.sp
    def __init__(self, Es, k=2, device=None):
        assert k == 2
        *_, A, X = Es.shape
        assert A == X ** (k-1)
        self.E = torch.full((A, 2, X), self.div(1.0, X), device=device)
        self.E[:, 1, :] = Es # shape AQX Q=2        
        self.k = k        
        self.device = device
        assert k == 2

    def update_states(self, states, segment, need_copy=True):
        if need_copy: # need to do this to get gradients properly
            states = states.clone() 
        states[segment] = 1
        return states

    def logp_symbol_sequence(self, sequence, starting_state=INITIAL_STATE_INDEX):
        num_automata = self.E.shape[-3]
        states = torch.LongTensor([starting_state for _ in range(num_automata)]) # shape A
        states[BOUNDARY_SYMBOL_INDEX] = 1
        states = states.unsqueeze(-1).repeat(1, self.num_symbols).unsqueeze(-2).to(self.device)
        logp = 0.0
        for segment in sequence:
            # p(seg|states) = 1/Z \prod_a p_a(seg|states[a]) # E has shape A x Q x X
            sub_probs = torch.gather(self.E, -2, states).squeeze(-2)
            log_numerators = self.to_log(sub_probs).sum(-2) # shape X
            logZ = log_numerators.logsumexp(-1) # scalar
            logp += log_numerators[segment] - logZ
            # perform SP update on the states -- if a segment has been seen,
            # update all states relevant for that segment.
            states = self.update_states(states, segment, need_copy=True)
        return logp

    def generate(self, starting_state=INITIAL_STATE_INDEX):
        num_automata = self.E.shape[-3]
        states = torch.LongTensor([starting_state for _ in range(num_automata)]) # shape A
        states = states.unsqueeze(-1).repeat(1, self.num_symbols).unsqueeze(-2)
        while True: 
            sub_probs = torch.gather(self.E, -2, states).squeeze(-2)
            log_numerators = self.to_log(sub_probs).sum(-2) # shape X, LOG SPACE
            logZ = log_numerators.logsumexp(-1)
            logprobs = log_numerators - logZ
            probs = np.exp(logprobs.to('cpu').detach().numpy())
            x = np.random.choice(self.inventory, p=probs)
            yield (None, x)
            if x == BOUNDARY_SYMBOL_INDEX:
                break
            states = update_states(states, segment, need_copy=False)


# Strictly k-local: Log probability only depends (k-1) preceding segments
# I[ x_t : x_1, ... x_{t-k} | x_{t-k+1, ..., t-1} ] = 0

# x_1, x_2, x_3, x_4, x_5
# I[ x_4 : x_1, x_2 | x_3 ] = 0

class Logspace:
    """ storing and manipulating log probabilities whenever possible """
    zero = -INF
    one = 0
    mul = torch.add
    sum = torch.logsumexp
    div = torch.sub
    to_exp = torch.exp

    def to_log(self, x):
        return x

    if torch.cuda.is_available() and genbmm is not None:
        def matmul(self, log_A, log_B):
            if log_A.ndim == 1:
                log_A = log_A.unsqueeze(0)
            log_A = log_A.unsqueeze(0)
            log_B = log_B.unsqueeze(0)
            return genbmm.logbmm(log_A, log_B)
    else:
        def matmul(self, log_A, log_B):
            #     Log-space matrix multiplication operator
            #     log_A : m x n
            #     log_B : n x p
            #     output : m x p matrix

            #     Normally, a matrix multiplication
            #     computes out_{i,j} = sum_k A_{i,k} x B_{k,j}
            
            #     A log domain matrix multiplication
            #     computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}
            #     """
            if log_A.ndim == 1:
                log_A_expanded = log_A.unsqueeze(0).unsqueeze(-1) # m x n x 1      # if log_A is a vector, 1 x n x 1
            else:
                log_A_expanded = log_A.unsqueeze(-1)
            log_B_expanded = log_B.unsqueeze(-3) # 1 x n x p      
            result = (log_A_expanded + log_B_expanded).logsumexp(dim=-2) # m x p
            if log_A.ndim == 1:
                return result.squeeze(-2)
            else:
                return result

    def entropy(self, logps, eps=EPSILON):
        ps = logps.exp()
        return -(ps * logps).sum() / LOG2

    @property
    def stationary_Q(self, eps=EPSILON):
        """ Vector of length q giving the stationary logp(q) """
        pspace_stationary = stationary(self.to_exp(self.state_transition_matrix))
        return (pspace_stationary + eps).log()


class LogspaceSimplePFA(SimplePFA, Logspace):
    pass

def string_power(V, k):
    return itertools.product(*[range(V) for _ in range(k)])

def strictly_local_transition_matrix(inventory_size, k=2):
    """ Transition matrix for an SL-k automaton """
    T_shape = (inventory_size,)*(k+1)
    T = torch.zeros(T_shape)
    inventory = range(inventory_size)
    for *context, last in string_power(inventory_size, k - 1):
        index = (...,) + tuple(context) + (last, last)
        T[index] = 1
    return T.reshape(inventory_size**(k-1), inventory_size, inventory_size)
    
def gradient_descent(model_class,
                     num_symbols,
                     training_data,
                     num_epochs=DEFAULT_NUM_EPOCHS,
                     nondeterminism_penalty=0,
                     memory_mi_penalty=0,
                     yield_every=DEFAULT_PRINT_EVERY,
                     batch_size=DEFAULT_BATCH_SIZE,
                     init_temperature=DEFAULT_INIT_TEMPERATURE,
                     activation=DEFAULT_ACTIVATION, 
                     **kwds):
    """ Given a PFA class and training data, return a PFA which has been trained by gradient descent. 
    Inputs:
    - model class: If an integer, then it's a PFA with that number of states.
    If it's a string "sp" or "sl", then it's strictly piecewise or strictly local.

    - num_symbols: The size of the symbol inventory, including the EOS symbol.

    - training_data: A sequence of sequences of integers, including the index for EOS at the end of each sequence.

    - num_epochs: Number of gradient descent steps.

    - nondeterminism_penalty: Regularization parameter for nondeterminism in the transition matrix.
    - memory_mi_penalty: Regularization parameter for memory MI of the PFA.
    
    - yield_every: Write intermediate results every x epochs. None: only write at end.

    Output:
    A series of PFA objects.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_states = model_class if isinstance(model_class, int) else None
    
    if activation == "softmax":
        activation = torch.softmax
    elif activation == "sparsemax":
        activation = entmax.sparsemax
    elif activation == "entmax15":
        activation = entmax.entmax15
    else:
        raise ValueError("activation must one one of {softmax, sparsemax, entmax15}")
    
    assert all(sequence[-1] == BOUNDARY_SYMBOL_INDEX for sequence in training_data)            
    if num_states:
        # If we're parameterizing by number of states, we need a way to enforce behavior around the boundary
        # To do that, we define an affine transformation T' = aT + b, such that T' respects boundaries
        # by going into the "initial" state deterministically after seeing the boundary symbol.
        mask = torch.ones(num_states, num_symbols, num_states).to(device)
        mask[:, BOUNDARY_SYMBOL_INDEX, :] = 0 # zero out all transitions from the boundary symbol...
        to_add = torch.zeros(*mask.shape).to(device)
        to_add[:, BOUNDARY_SYMBOL_INDEX, INITIAL_STATE_INDEX] = 1 # ...except for going into the initial state
        # Now we can do T' = mask*T + to_add to make T' respect boundaries

    if num_states:
        E_logit = 1/init_temperature*torch.randn(num_states, num_symbols)
        T_logit = 1/init_temperature*torch.randn(num_states, num_symbols, num_states)        
        E_logit = E_logit.to(device).detach().requires_grad_(True)
        T_logit = T_logit.to(device).detach().requires_grad_(True)
        params = [E_logit, T_logit]
    elif model_class.startswith('sl'):
        if model_class == "sl":
            k = 2
        else:
            k = int(model_class.replace("sl", ""))
        E_logit = 1/init_temperature*torch.randn(num_symbols, num_symbols)
        E_logit = E_logit.to(device).detach().requires_grad_(True)
        T_logit = None
        T = strictly_local_transition_matrix(num_symbols, k=k).to(device)        
        params = [E_logit]
    elif model_class.startswith('sp_sl'):
        E_logit = 1/init_temperature*torch.randn(2, num_symbols, num_symbols)
        E_logit = E_logit.to(device).detach().requires_grad_(True)
        T_logit = None
        params = [E_logit]
    elif model_class == 'sp':
        E_logit = 1/init_temperature*torch.randn(num_symbols, num_symbols)
        E_logit = E_logit.to(device).detach().requires_grad_(True)
        T_logit = None
        params = [E_logit]

    opt = torch.optim.Adam(params=params, **kwds)
    for i in range(num_epochs):
        opt.zero_grad()

        # Create E and T matrices from underlying parameters
        E = activation(E_logit, dim=-1)
        if T_logit is not None:
            T = activation(T_logit, dim=-1) * mask + to_add # convert to probability and handle word boundary behavior

        # Initialize PFA
        if model_class == 'sp':
            pfa = FastStrictlyPiecewisePDFA(E, device=device)
        elif model_class == 'sp_sl':
            pfa = HomogenousFactorPDFA.sp_sl(E, device=device)
        elif model_class == 'sp_sl_slow':
            pfa = sp_sl_slow(E[0], E[1], device=device)
        else:
            pfa = SimplePFA(E, T, device=device)

        # Get batch of training data
        if batch_size:
            this_batch = random.sample(training_data, batch_size) 
        else:
            this_batch = training_data

        # Get NLL
        nll = 0
        for seq in this_batch:
            nll -= pfa.logp_symbol_sequence(seq)

        # Get information lattice quantities
        if nondeterminism_penalty == 0 and memory_mi_penalty == 0: # fast path to avoid calculating stationary distribution if not necessary
            info_penalties = 0 
        else:
            info_penalties = nondeterminism_penalty * pfa.nondeterminism + memory_mi_penalty * pfa.memory_mi
            
        loss = nll + info_penalties

        # Differentiate and optimize
        loss.backward()
        opt.step()

        # Yield intermediate results
        if yield_every is not None and i % yield_every == 0:
            yield i, pfa

    yield i, pfa


def star_a_dots_b_example(p_halt=.1):
    """ Generates a 2-SP language *a...b """
    # *a...b
    # so allow bbbaaaaaa# etc. segments are {a,b,c}
    # state 1: boundary, from which we can generate a or b or c or boundary
    # state 2: seen a, from which we can generate a or c boundary
    E = torch.Tensor([
        [p_halt, (1-p_halt)/3, (1-p_halt)/3, (1-p_halt)/3],
        [p_halt, (1-p_halt)/2, 0, (1-p_halt)/2]
    ])
    T = torch.Tensor([
        # from state 0, on action a, go into state 1, otherwise stay in state 0
        [[1, 0], # on action #
         [0, 1], # on action a
         [1, 0], # on action b
         [1, 0]], # on action c
        # from state 1, on action #, go into state 0, otherwise stay
        [[1, 0], # on action #
         [0, 1], # on action a
         [0, 1], # on action b (irrelevant)
         [0, 1]], # on action c
    ])
    return SimplePDFA(E, T)

def star_ab_example(p_halt=.25):
    """ Generates a 2-SL language *ab """
    # two memory states: q0 and q1
    # four symbols: #, a, b, c
    # The language is *ab, so we generate things like #bbbbaaa#aaa#baaaaaaa##b#
    E = np.array([
        [p_halt, (1-p_halt)/3, (1-p_halt)/3, (1-p_halt)/3],  # from state 0, we generate #, a, or b
        [p_halt, (1-p_halt)/2, 0, (1-p_halt)/2]  # from state 1, we generate # or a, never b
    ])
    T = np.array([
        # on action #, from state 0, we go into state 0, and from state 1, we go into state 0
        [[1, 0], [1, 0]],
        # on action a, from state 0, we go into state 1, and from state 1, we go into state 1
        [[0, 1], [0, 1]],
        # on action b, from state 0, we go into state 0, and from state 1, undefined behavior
        [[1, 0], [1, 0]],
        # on action c, go to state 0
        [[1, 0], [1, 0]],
    ]).transpose(1,0,2)

    ab = SimplePDFA(torch.Tensor(E), torch.Tensor(T))
    return ab

def some_ab_example(p_halt=.25):
    """ Generates a 2-LT language \exists ab """
    E = torch.Tensor([
        [0, 1/3, 1/3, 1/3], # from state 0, generate {a,b,c}
        [0, 1/3, 1/3, 1/3], # from state 1, generate {a,b,c]
        [p_halt, (1-p_halt)/3, (1-p_halt)/3, (1-p_halt)/3],
    ])
    T = torch.Tensor([
        # in state 0,
        [[1, 0, 0],  # upon seeing #
         [0, 1, 0],  # upon seeing a
         [1, 0, 0],  # upon seeing b
         [1, 0, 0]],  # upon seeing c.
        # in state 1,
        [[1, 0, 0], # upon seeing #
         [0, 1, 0], # upon seeing a
         [0, 0, 1], # upon seeing b
         [1, 0, 0]], # upon seeing c
        # in state 2,
        [[1, 0, 0], # upon seeing #
         [0, 0, 1], # upon seeing a
         [0, 0, 1], # upon seeing b
         [0, 0, 1]], # upon seeing c
    ])
    return SimplePDFA(E, T)

def some_a_dots_b_example(p_halt=.25):
    """ Generates a 2-PT language \exists a...b """
    E = torch.Tensor([
        [0, 1/3, 1/3, 1/3], # from state 0, generate {a,b,c}
        [0, 1/3, 1/3, 1/3], # from state 1, generate {a,b,c]
        [p_halt, (1-p_halt)/3, (1-p_halt)/3, (1-p_halt)/3],
    ])
    T = torch.Tensor([
        # in state 0,
        [[1, 0, 0],  # upon seeing #
         [0, 1, 0],  # upon seeing a
         [1, 0, 0],  # upon seeing b
         [1, 0, 0]],  # upon seeing c.
        # in state 1,
        [[1, 0, 0], # upon seeing #
         [0, 1, 0], # upon seeing a
         [0, 0, 1], # upon seeing b
         [0, 1, 0]], # upon seeing c
        # in state 2,
        [[1, 0, 0], # upon seeing #
         [0, 0, 1], # upon seeing a
         [0, 0, 1], # upon seeing b
         [0, 0, 1]], # upon seeing c
    ])
    return SimplePDFA(E, T)

def one_ab_example(p_halt=.25):
    """ Generates a 2-LTT language one-ab """
    E = torch.Tensor([
        [0, 1/3, 1/3, 1/3], # initial state
        [0, 1/3, 1/3, 1/3], # seen a
        [p_halt, (1-p_halt)/3, (1-p_halt)/3, (1-p_halt)/3], # seen ab...
        [p_halt, (1-p_halt)/2, 0, (1-p_halt)/2], # seen ab...a
    ])
    T = torch.Tensor([
        # in state 0,
        [[1, 0, 0, 0],  # upon seeing # (never happens)
         [0, 1, 0, 0],  # upon seeing a -> go to state seen a
         [1, 0, 0, 0],  # upon seeing b
         [1, 0, 0, 0]],  # upon seeing c.
        # in state seen a,
        [[1, 0, 0, 0], # upon seeing # (never happens)
         [0, 1, 0, 0], # upon seeing a, stay 
         [0, 0, 1, 0], # upon seeing b, go to state seen ab
         [1, 0, 0, 0]], # upon seeing c, revert to initial
        # in state 2,
        [[1, 0, 0, 0], # upon seeing #
         [0, 0, 0, 1], # upon seeing a, go to state seen ab...a
         [0, 0, 1, 0], # upon seeing b, stay
         [0, 0, 1, 0]], # upon seeing c, stay
        # in state 3,
        [[1, 0, 0, 0], # upon seeing #
         [0, 0, 0, 1], # upon seeing a, stay in state ab...a
         [0, 0, 1, 0], # upon seeing b (never happens)
         [0, 0, 1, 0]], # upon seeing c, go to state seen ab...
    ])
    return SimplePDFA(E, T)
   

def learn_example(pfa, legal_examples, illegal_examples, model_class="pfa", num_samples=10000, num_epochs=10000, outfile=sys.stderr, print_every=1, **kwds):
    if model_class == "pfa":
        model_class = pfa.num_states
    data = [tuple(x for q, x in pfa.generate()) for _ in range(num_samples)]
    writer = csv.writer(outfile)
    writer.writerow(['epoch','model_class','legal_illegal_diff'])
    for i, ahat in gradient_descent(model_class, pfa.num_symbols, data, num_epochs=num_epochs, yield_every=print_every):
        legal_ll = sum(ahat.logp_symbol_sequence(legal_example).item() for legal_example in legal_examples)
        illegal_ll = sum(ahat.logp_symbol_sequence(illegal_example).item() for illegal_example in illegal_examples)
        writer.writerow([i, model_class, legal_ll - illegal_ll])
    return ahat

def learn_some_ab_example(p_halt=.1, **kwds):
    a = some_ab_example(p_halt)
    legal = [(2, 1, 2, 3, 3, 3, 0)] # babccc#
    illegal = [(2, 1, 3, 3, 3, 2, 0)] # bacccb#
    return learn_example(a, legal, illegal, **kwds)

def learn_some_a_dots_b_example(p_halt=.1, **kwds):
    a = some_a_dots_b_example(p_halt)
    legal = [(2, 1, 3, 3, 3, 2, 0)] # bacccb#
    illegal = [(2, 2, 1, 3, 3, 3, 0)] # bbaccc#
    return learn_example(a, legal, illegal, **kwds)

def learn_star_ab_example(p_halt=.1, **kwds):
    a = star_ab_example(p_halt)
    legal =   [(2, 1, 3, 3, 3, 2, 0)] # bacccb#
    illegal = [(2, 1, 2, 3, 3, 3, 0)] # babccc#
    return learn_example(a, legal, illegal, **kwds)

def learn_star_a_dots_b_example(p_halt=.1, **kwds):
    a = star_a_dots_b_example(p_halt)
    legal =   [(2, 1, 3, 3, 3, 1, 0)] # baccca#
    illegal = [(2, 1, 3, 3, 3, 2, 0)] # bacccb#
    return learn_example(a, legal, illegal, **kwds)

def star_ab_tests(ab, p_halt, eps=10**-6):
    start = 0
    assert np.abs(np.exp(ab.logp_symbol_sequence([0], starting_state=start)) - p_halt) < eps # p(#)
    assert np.abs(np.exp(ab.logp_symbol_sequence([0,0], starting_state=start)) - p_halt**2) < eps # p(##)
    assert np.abs(np.exp(ab.logp_symbol_sequence([1,0], starting_state=start)) - (1-p_halt)/3 * p_halt) < eps # p(a#)
    assert np.abs(np.exp(ab.logp_symbol_sequence([2,0], starting_state=start)) - (1-p_halt)/3 * p_halt) < eps # p(b#)
    assert np.abs(np.exp(ab.logp_symbol_sequence([2,1,0], starting_state=start)) - ((1-p_halt)/3)**2 * p_halt) < eps # p(ba#)
    assert np.abs(np.exp(ab.logp_symbol_sequence([1,2,0], starting_state=start)) - 0) < eps # p(ab#) = 0
    assert np.abs(np.exp(ab.logp_symbol_sequence([1,1,2,0], starting_state=start)) - 0) < eps # p(aab#) = 0
    assert np.abs(np.exp(ab.logp_symbol_sequence([2,1,2,0], starting_state=start)) - 0) < eps # p(aab#) = 0
    assert np.abs(np.exp(ab.logp_symbol_sequence([1,1,1,0], starting_state=start)) - ((1-p_halt)/3)*((1-p_halt)/2)**2 * p_halt) < eps # p(aaa#)
    assert np.abs(np.exp(ab.logp_symbol_sequence([2,1,1,0], starting_state=start)) - ((1-p_halt)/3)**2 * ((1-p_halt)/2)**1 * p_halt) < eps # p(baa#) = 0
    
def test_star_ab_example(pkg='torch'):
    p_halt = .25
    ab = star_ab_example(p_halt)
    star_ab_tests(ab, p_halt)

def test_logspace_star_ab_example():
    p_halt = .25
    ab = star_ab_example(p_halt)
    ab = LogspaceSimplePFA((ab.E + EPSILON).log(), (ab.T + EPSILON).log())
    star_ab_tests(ab, p_halt)


def get_txt_corpus_data(filename):
    """
    Reads input file and coverts it to list of lists. Word boundaries will be added later.
    """
    with open(filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            yield line.strip().split(' ')

def shuffled(xs):
    xs = list(xs)
    random.shuffle(xs)
    return xs

def evaluate(testfile, model, phone2ix, num_samples=DEFAULT_PERM_TEST_NUM_SAMPLES):

    def score_form(form):
        indices = [phone2ix[p] for p in form.split(' ')] + [BOUNDARY_SYMBOL_INDEX]
        return model.logp_symbol_sequence(indices).item()
    
    d = pd.read_csv(testfile, sep="\t", header=None)
    d.columns = ['form', 'legality']
    d['legality'] = d['legality'].map(lambda x: x.split('-')[0])
    d['score'] = d['form'].map(score_form)
    real_result = d.groupby(['legality']).mean().reset_index()
    real_difference = real_result['score'].diff()[1]

    # Permutation test
    if not num_samples:
        return d, None
    else:
        fake_differences = []
        for _ in range(num_samples):
            d['fake_legality'] = shuffled(d['legality'])
            fake_result = d.groupby(['fake_legality']).mean().reset_index()
            fake_difference = fake_result['score'].diff()[1]
            fake_differences.append(fake_difference)
        fake_differences.append(real_difference)
        fake_differences.sort()
        p_value = fake_differences.index(real_difference) / num_samples
        return d, p_value

def process_data(string_training_data, dev=True, training_split=.2):
    # EOS is always index 0
    inventory = [EOS] + list(set(phone for w in string_training_data for phone in w))

    # dictionaries for looking up the index of a phone and vice versa
    phone2ix = {p: ix for (ix, p) in enumerate(inventory)}
    ix2phone = {ix: p for (ix, p) in enumerate(inventory)}

    as_ixs = [
        torch.LongTensor([phone2ix[p] for p in sequence] + [BOUNDARY_SYMBOL_INDEX])
        for sequence in string_training_data
    ]

    if not dev:
        training_data = as_ixs  
        dev = []
    else:
        split = int(len(as_ixs) * (1 - training_split))
        training_data = as_ixs[:split] 
        dev = as_ixs[split:] 

    return phone2ix, ix2phone, training_data, dev

def main(input_file,
         test_file,
         model_class=DEFAULT_NUM_STATES,
         num_epochs=DEFAULT_NUM_EPOCHS,
         num_samples=DEFAULT_NUM_SAMPLES,
         print_every=DEFAULT_PRINT_EVERY,
         seed=DEFAULT_DATA_SEED,
         perm_test_num_samples=DEFAULT_PERM_TEST_NUM_SAMPLES,
         **kwds):
    # model_class is either 'sp', 'sl', or a number of states
    first_line = True
    random.seed(seed)
    data = shuffled(get_txt_corpus_data(input_file))
    phone2ix, ix2phone, training_data, dev = process_data(data)
    num_symbols = len(ix2phone)
    d = kwds.copy()
    print("Training set size =", len(training_data), file=sys.stderr)
    print("Dev set size =", len(dev), file=sys.stderr)
    print("Segment inventory size = ", len(ix2phone), file=sys.stderr)
    if model_class is None:
        model_class = len(ix2phone)*2
    d['model_class'] = model_class
    print("Model class =", model_class, file=sys.stderr)
    for epoch, a in gradient_descent(model_class, num_symbols, training_data, num_epochs=num_epochs, yield_every=print_every, **kwds):
        print(epoch, file=sys.stderr)
        d['epoch'] = epoch
        d['train_nll'] = -torch.stack([
            a.logp_symbol_sequence(seq)
            for seq in training_data
        ]).mean().item()
        d['dev_nll'] = -torch.stack([
            a.logp_symbol_sequence(seq)
            for seq in dev
        ]).mean().item()
        d['nondeterminism'] = a.nondeterminism.item()
        d['memory_mi'] = a.memory_mi.item()
        for j in range(num_samples):
            so_far = []
            for s, i in a.generate():
                if i != BOUNDARY_SYMBOL_INDEX:
                    so_far.append(i)
                else:
                    phones = "".join(ix2phone[i] for i in so_far)
                    d['sample_%d' % j] = phones
                    break
        nonce_result, p_value = evaluate(test_file, a, phone2ix, num_samples=perm_test_num_samples)
        illegal, legal = nonce_result.groupby(['legality']).mean().reset_index()['score']
        d['legal_nll'] = -legal
        d['illegal_nll'] = -illegal
        d['p_value'] = p_value
        if first_line:
            writer = csv.DictWriter(sys.stdout, fieldnames=d.keys())
            writer.writeheader()
            first_line = False
        writer.writerow(d)
    return a, ix2phone

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Induce and evaluate a PFA to model forms.')
    parser.add_argument("--lang", type=str, default="quechua", help="language data to use (by default Quechua)")
    parser.add_argument("--model_class", type=str, default=DEFAULT_NUM_STATES, help="model class (integer, sp, sl, sp_sl, or slk for natural number k)")
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--nondeterminism_penalty", type=float, default=0.0)
    parser.add_argument("--memory_mi_penalty", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="batch size; 0 means full gradient descent with no batches")
    parser.add_argument("--init_temperature", type=float, default=DEFAULT_INIT_TEMPERATURE, help="initialization temperature")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="starting learning rate for Adam")
    parser.add_argument("--activation", type=str, default=DEFAULT_ACTIVATION, help="activation function for probabilities: softmax, sparsemax, or entmax15")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES, help="number of samples to output")
    parser.add_argument("--perm_test_num_samples", type=int, default=DEFAULT_PERM_TEST_NUM_SAMPLES, help="number of samples in permutation test")
    parser.add_argument("--print_every", type=int, default=DEFAULT_PRINT_EVERY, help="print results per x epochs")
    parser.add_argument("--seed", type=int, default=DEFAULT_DATA_SEED, help="random seed for train-dev-test split and batches")
    args = parser.parse_args()
    try:
        model_class = int(args.model_class)
    except ValueError:
        model_class = args.model_class
    if args.lang == "quechua":
        learning_filename = "LearningData.txt"
        testing_filename = "TestingData.txt"
    else:
        learning_filename = "LearningData_%s.txt" % args.lang
        testing_filename = "TestingData_%s.txt" % args.lang
    main(learning_filename,
         testing_filename,
         model_class=model_class,
         num_epochs=args.num_epochs,
         nondeterminism_penalty=args.nondeterminism_penalty,
         memory_mi_penalty=args.memory_mi_penalty,
         init_temperature=args.init_temperature,
         activation=args.activation,
         batch_size=args.batch_size,
         lr=args.lr,
         num_samples=args.num_samples,
         print_every=args.print_every,
         seed=args.seed,
         perm_test_num_samples=args.perm_test_num_samples)

