import sys

def matrix_mult(A, B, mod):
    size = len(A)
    C = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            for k in range(size):
                C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % mod
    return C

def matrix_pow(A, p, mod):
    size = len(A)
    I = [[0] * size for _ in range(size)]
    for i in range(size):
        I[i][i] = 1
    if p == 0:
        return I
    if p % 2 == 1:
        return matrix_mult(A, matrix_pow(A, p - 1, mod), mod)
    X = matrix_pow(A, p // 2, mod)
    return matrix_mult(X, X, mod)

MOD = 998244353

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    additional_edges = []
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        additional_edges.append((u, v))

    # Vertices in S_1based
    S_1based = set([1])
    for u, v in additional_edges:
        S_1based.add(u)
        S_1based.add(v)

    S_list = sorted(list(S_1based))
    m = len(S_list)
    s_to_idx = {s: i for i, s in enumerate(S_list)}

    # Compute cyclic arc lengths D_i for s_i -> s_{i+1} in S_list
    D = []
    for i in range(m):
        s_curr = S_list[i]
        s_next = S_list[(i + 1) % m]
        dist = (s_next - s_curr + N) % N
        if dist == 0: # m=1 implies dist is N
             D.append(N)
        else:
            D.append(dist)

    # State vector layout:
    # (f[k][s_i])_{i=0}^{m-1} : indices 0 to m-1
    # (H[k][s_i])_{i=0}^{m-1} : indices m to 2m-1
    # (f[k-D_i][s_i])_{i=0}^{m-1, D_i > 1} : indices 2m onwards
    # Identify indices i where D_i > 1
    idx_D_gt_1 = [i for i in range(m) if D[i] > 1]

    # Map (D_i, i) to state index for f_past values
    f_past_state_map = {}
    current_state_idx = 2 * m
    for i in idx_D_gt_1:
        f_past_state_map[(D[i], i)] = current_state_idx
        current_state_idx += 1

    state_size = current_state_idx
    Trans = [[0] * state_size for _ in range(state_size)]

    # Transitions for f[k][s_j] (state index j, where j = s_to_idx[s_j])
    for j in range(m):
        s_j = S_list[j]
        
        # Additional edges sum: sum A_{ij} f[k-1][s_i]
        # Contribution from f[k-1][s_i] (which is f[k][s_i] in prev step, index i)
        for u, v in additional_edges:
            if v == s_j:
                s_i = u
                i = s_to_idx[s_i]
                Trans[j][i] = (Trans[j][i] + 1) % MOD
        
        # Cyclic edge term: f[k-D_{p_j}][s_{p_j}]
        # p_j_idx is index of vertex in S_list before s_j cyclically
        p_j_idx = (j - 1 + m) % m
        D_p_j = D[p_j_idx]
        s_p_j = S_list[p_j_idx]
        
        if D_p_j == 1:
            # Term is f[k-1][s_{p_j}]. Comes from f[k][s_{p_j}] in previous step (index p_j_idx)
            Trans[j][p_j_idx] = (Trans[j][p_j_idx] + 1) % MOD
        else:
            # Term is f[k-D_{p_j}][s_{p_j}]. Comes from f[k-D_{p_j}][s_{p_j}] state in previous step
            Trans[j][f_past_state_map[(D_p_j, p_j_idx)]] = (Trans[j][f_past_state_map[(D_p_j, p_j_idx)]] + 1) % MOD

    # Transitions for H[k][s_j] (state index m+j)
    # H[k][s_j] = f[k-1][s_j] + H[k-1][s_j] - f[k-D_j][s_j] (if D_j > 1)
    for j in range(m):
        if D[j] > 1:
            # f[k-1][s_j] contribution comes from f[k][s_j] in prev state (index j)
            Trans[m + j][j] = (Trans[m + j][j] + 1) % MOD
            # H[k-1][s_j] contribution comes from H[k][s_j] in prev state (index m+j)
            Trans[m + j][m + j] = (Trans[m + j][m + j] + 1) % MOD
            # -f[k-D_j][s_j] contribution comes from f[k-D_j][s_j] state in prev state (index f_past_state_map[(D_j, j)])
            Trans[m + j][f_past_state_map[(D[j], j)]] = (Trans[m + j][f_past_state_map[(D[j], j)]] - 1 + MOD) % MOD
        # If D_j = 1, H[k][s_j] = 0, so no transitions from previous H or f_past

    # Transitions for f[k-D_i][s_i] for i in idx_D_gt_1 (state index f_past_state_map[(D_i, i)])
    # Let d = D_i. State holds f[k-d][s_i] at step k. At step k+1 it holds f[k+1-d][s_i].
    # The value f[k+1-d][s_i] at step k+1 is the value f[k+1-d][s_i] computed at step k.
    # Where was f[k+1-d][s_i] stored at step k?
    for i in idx_D_gt_1:
        d = D[i]
        d_prime = d - 1 # The step difference from k
        
        if d_prime == 0: # This implies d=1, but this loop is for D_i > 1
             pass # Should not happen
        elif d_prime == 1: # d = 2. Value f[k-1][s_i] at step k. This is f[k][s_i] in prev step (index i)
            Trans[f_past_state_map[(d, i)]][i] = (Trans[f_past_state_map[(d, i)]][i] + 1) % MOD
        elif d_prime > 1: # d > 2. Value f[k-(d-1)][s_i] at step k.
            # This value is stored in the f_past state variable for (d-1, i) IF D_i = d-1 > 1.
            # But D_i is d, not d-1.
            # The state variable f[k-d][s_i] gets its value from f[k-(d-1)][s_i] from the previous time step.
            # The value f[k-(d-1)][s_i] was stored somewhere in the state vector AT THE PREVIOUS STEP.
            # If d-1 == 1: It was f[k-1][s_i], which was at index i in prev state.
            # If d-1 > 1: It was f[k-(d-1)][s_i], which was at index f_past_state_map[(d-1, i)] in prev state.
            Trans[f_past_state_map[(d, i)]][f_past_state_map[(d_prime, i)]] = (Trans[f_past_state_map[(d, i)]][f_past_state_map[(d_prime, i)]] + 1) % MOD


    # Initial state vector (at k=0)
    # f[0][1] = 1, others 0. H[0] = 0. f[-D_i] = 0.
    initial_state = [0] * state_size
    initial_state[s_to_idx[1]] = 1 # f[0][1]

    # Matrix exponentiation
    if K > 0:
        Trans_K = matrix_pow(Trans, K, MOD)
        final_state_vec = [0] * state_size
        for i in range(state_size):
            for j in range(state_size):
                final_state_vec[i] = (final_state_vec[i] + Trans_K[i][j] * initial_state[j]) % MOD

        # Total ways = sum_{i=0}^{m-1} (f[K][s_i] + H[K][s_i])
        total_ways = 0
        for i in range(m):
            total_ways = (total_ways + final_state_vec[i]) % MOD # f[K][s_i]
            total_ways = (total_ways + final_state_vec[m + i]) % MOD # H[K][s_i]

    else: # K=0
        total_ways = 1 # Start at vertex 1, 0 steps. Sequence (1).

    print(total_ways)

solve()