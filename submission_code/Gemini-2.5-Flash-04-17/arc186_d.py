import sys

sys.setrecursionlimit(300500)
MOD = 998244353

N = int(sys.stdin.readline())
A = list(map(int, sys.stdin.readline().split()))

# dp[i][k][f]: number of ways to complete V_i .. V_{N-1}
# i: current index to fill (0 to N)
# k: needed count *before* choosing V_i (needed_{i-1})
# f: is_less flag for prefix V_0..V_{i-1} vs A_0..A_{i-1}

# Base case: i == N (all elements V_0..V_{N-1} are chosen)
# k is the needed count before processing V_N (which doesn't exist).
# This k represents needed_{N-1}. Needed_{N-1} must be 0 for a Polish sequence of length N.
# So, if i == N, return 1 if k == 0, else 0.

# State dimensions:
# i: 0 to N
# k: needed count before V_i. k = needed_{i-1}.
#    needed_j >= 1 for j=0..N-2. needed_{N-1}=0.
#    needed_{i-1} can range from 1 up to N-(i-1) = N-i+1. Max k is N+1 (when i=0, N=0).
#    For i=0, k=needed_{-1}=1. Range [1, 1].
#    For i=1, k=needed_0. $needed_0 = needed_{-1}-1+V_0 = 1-1+V_0=V_0$.
#    $V_0$ can range up to N-1. But $needed_0 \le N-1$.
#    Correct needed_j bound: $1 \le needed_j \le N-j$ for $j < N-1$. needed_{N-1}=0.
#    So needed_{i-1} range is $[1, N-(i-1)] = [1, N-i+1]$ for $i-1 < N-1 \implies i < N$.
#    For $i=N$, k=needed_{N-1}=0.
#    Max k needed is N-0+1=N+1 for i=0? No. Max needed_{i-1} is N-(i-1).
#    Max needed is N at index 0? N-0=N.
#    The needed count at index j (after V_j) is needed_j. Max needed_j is N-j.
#    So needed_{i-1} max is N-(i-1) = N-i+1.
#    k ranges from 1 to N-i+1 for i <= N-1. For i=N, k=0.
#    Max k is N+1 at i=0 if N=0? No. Max k at i=0 is $N-0+1 = N+1$. But needed_{-1}=1.
#    Max k overall is N.

# Let's use DP table size (N+1) x (N+2).
dp_f = [[0] * (N + 2) for _ in range(N + 1)]
dp_g = [[0] * (N + 2) for _ in range(N + 1)]

# Base case i=N. k is needed_{N-1}.
# needed_{N-1}=0. So only dp[N][0][f] is relevant.
# dp[N][0][0] = 1, dp[N][0][1] = 1.
# States k > 0 for i=N should not be reached as needed_{N-1} must be 0.
dp_f[N][0] = 1
dp_g[N][0] = 1

for i in range(N - 1, -1, -1):
    # Sg[j], Sf[j]: prefix sums for dp[i+1] layer, up to needed count j.
    # needed_i = k_prev - 1 + V_i. This is the 'k' value for dp[i+1][k].
    # needed_i must be in range [0, N-(i+1)] if i < N-1, [0] if i=N-1.
    # max needed_i is N-(i+1).
    max_needed_next_step = N - (i + 1) # for i < N-1
    if i == N-1: max_needed_next_step = 0 # needed_{N-1} = 0

    # Sg, Sf arrays store prefix sums up to max_needed_next_step.
    # Indices 0 to max_needed_next_step. Size max_needed_next_step + 1.
    Sg = [0] * (max_needed_next_step + 1)
    Sf = [0] * (max_needed_next_step + 1)

    if max_needed_next_step >= 0:
        Sg[0] = dp_g[i + 1][0]
        Sf[0] = dp_f[i + 1][0]
        for j in range(1, max_needed_next_step + 1):
             Sg[j] = (Sg[j - 1] + dp_g[i + 1][j]) % MOD
             Sf[j] = (Sf[j - 1] + dp_f[i + 1][j]) % MOD

    # k_prev: needed count before V_i. Ranges from 1 to N-i+1.
    for k_prev in range(1, N - i + 2):
        # If k_prev == 0 or k_prev > N-i+1, invalid state, loop skips.

        # Range for V_i = v
        # needed_i = k_prev - 1 + v.
        # needed_i must be >= 0.
        # needed_i must be >= 1 for i < N-1. => k_prev-1+v >= 1 => v >= 2-k_prev.
        # needed_i must be == 0 for i = N-1. => k_prev-1+v == 0 => v = 1-k_prev.
        
        v_min_actual = 0
        if i < N - 1:
            v_min_actual = max(0, 2-k_prev)
        elif i == N - 1:
             if k_prev != 1: # needed_{N-2} must be 1
                 continue
             v_min_actual = 0 # V_{N-1} must be 0

        # Upper bound for V_i (v)
        # Lexicographical bound: v <= (f ? N-1 : A[i])
        # Needed count bound: needed_i <= max_needed_next_step.
        # k_prev - 1 + v <= max_needed_next_step => v <= max_needed_next_step - (k_prev - 1).

        v_max_lex_g = N - 1
        v_max_lex_f = A[i]

        v_max_needed_bound = max_needed_next_step - (k_prev - 1)

        v_min = v_min_actual
        v_max_g = min(v_max_lex_g, v_max_needed_bound)
        v_max_f = min(v_max_lex_f, v_max_needed_bound)

        # Calculate dp_g[i][k_prev] (f=1 case)
        # Sum dp_g[i+1][k_prev-1+v] for v in [v_min, v_max_g]
        if v_min <= v_max_g:
             next_k_min = k_prev - 1 + v_min
             next_k_max = k_prev - 1 + v_max_g
             
             # Indices for Sg are 0..max_needed_next_step
             upper_idx = min(next_k_max, max_needed_next_step)
             lower_idx = next_k_min
             
             if lower_idx <= upper_idx:
                 dp_g[i][k_prev] = (Sg[upper_idx] - (Sg[lower_idx - 1] if lower_idx > 0 else 0) + MOD) % MOD
        
        # Calculate dp_f[i][k_prev] (f=0 case)
        # Sum dp_g[i+1][k_prev-1+v] for v in [v_min, min(A[i]-1, v_max_f)]
        v_max_f_g_sum = min(A[i] - 1, v_max_f)
        if v_min <= v_max_f_g_sum:
             next_k_min_f = k_prev - 1 + v_min
             next_k_max_f = k_prev - 1 + v_max_f_g_sum
             
             if next_k_min_f <= max_needed_next_step:
                 upper_idx = min(next_k_max_f, max_needed_next_step)
                 lower_idx = next_k_min_f
                 if lower_idx <= upper_idx:
                     dp_f[i][k_prev] = (Sg[upper_idx] - (Sg[lower_idx - 1] if lower_idx > 0 else 0) + MOD) % MOD

        # Add dp_f[i+1][k_prev-1+A[i]] if v=A[i] is possible
        if v_min <= A[i] and A[i] <= v_max_f:
            next_k_A_i = k_prev - 1 + A[i]
            if next_k_A_i <= max_needed_next_step:
                dp_f[i][k_prev] = (dp_f[i][k_prev] + dp_f[i+1][next_k_A_i]) % MOD

# Answer is dp[0][1][0].
# Index 0, needed count before V_0 is 1 (needed_{-1}=1), is_less is 0.
print(dp_f[0][1])