import sys

MOD = 998244353

def solve():
    N, M = map(int, sys.stdin.readline().split())
    B = list(map(int, sys.stdin.readline().split()))

    # G[i][j]: number of ways to fill B_0...B_{i-1} using values {j, ..., M}
    # i from 0 to N, j from 1 to M+1
    G = [[0] * (M + 2) for _ in range(N + 1)]
    for j in range(1, M + 2):
        G[0][j] = 1

    for i in range(1, N + 1):
        # B_{i-1} is the current element being processed
        for j in range(M, 0, -1):
            if B[i-1] != -1 and B[i-1] < j:
                G[i][j] = 0
            elif B[i-1] != -1 and B[i-1] >= j:
                # B_{i-1} is fixed to B[i-1] >= j. Need ways for B_0...B_{i-2} >= j.
                G[i][j] = G[i-1][j]
            else: # B[i-1] == -1
                # B_{i-1} can be any value >= j.
                # Ways using values >= j+1 for B_0...B_{i-1} is G[i][j+1].
                # Ways using value j for B_{i-1} and values >= j for B_0...B_{i-2} is G[i-1][j].
                G[i][j] = (G[i][j+1] + G[i-1][j]) % MOD
    
    # L[i][k]: number of ways to fill B_i...B_{N-1} using values {1, ..., k}
    # i from 0 to N, k from 0 to M
    L = [[0] * (M + 1) for _ in range(N + 1)]
    for k in range(M + 1):
        L[N][k] = 1

    for i in range(N - 1, -1, -1):
        # B_i is the current element being processed
        for k in range(1, M + 1):
            if B[i] != -1 and B[i] > k:
                L[i][k] = 0
            elif B[i] != -1 and B[i] <= k:
                # B_i is fixed to B[i] <= k. Need ways for B_{i+1}...B_{N-1} <= k.
                L[i][k] = L[i+1][k]
            else: # B[i] == -1
                # B_i can be any value <= k.
                # Ways using values <= k-1 for B_i...B_{N-1} is L[i][k-1].
                # Ways using value k for B_i and values <= k for B_{i+1}...B_{N-1} is L[i+1][k].
                L[i][k] = (L[i][k-1] + L[i+1][k]) % MOD

    total_sum = 0
    # v is 1-indexed vertex 1...N. v is the minimum in its component
    # <=> for all i < v and j >= v, B'_i > B'_j.
    # This is equivalent to min_{i in 0..v-2} B'_i > max_{j in v-1..N-1} B'_j.
    # Iterate over k = max_{j in v-1..N-1} B'_j. k from 1 to M.
    # If max_{j in v-1..N-1} B'_j = k, then min_{i in 0..v-2} B'_i >= k+1.
    # Ways = (ways B_0...B_{v-2} values >= k+1) * (ways B_{v-1}...B_{N-1} max value is k)
    for v in range(1, N + 1): # v is 1-indexed vertex
        v_split_idx = v - 1 # Split between B_0..B_{v-2} (length v-1) and B_{v-1}..B_{N-1} (start index v-1)
        
        for k in range(1, M + 1): # k is max value in suffix B_{v-1}..B_{N-1}, 1 <= k <= M
            # Ways for prefix B_0...B_{v-2} to have values >= k+1
            ways_prefix_geq_k_plus_1 = G[v_split_idx][k + 1]
            
            # Ways for suffix B_{v-1}...B_{N-1} to have max value exactly k
            ways_suffix_leq_k = L[v_split_idx][k]
            ways_suffix_leq_k_minus_1 = L[v_split_idx][k - 1]

            ways_suffix_max_k = (ways_suffix_leq_k - ways_suffix_leq_k_minus_1 + MOD) % MOD

            count_v_k = (ways_prefix_geq_k_plus_1 * ways_suffix_max_k) % MOD
            total_sum = (total_sum + count_v_k) % MOD

    print(total_sum)

solve()