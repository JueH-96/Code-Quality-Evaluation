import sys
import math

def solve():
    N = int(sys.stdin.readline())
    P_1based = list(map(int, sys.stdin.readline().split()))
    A_1based = list(map(int, sys.stdin.readline().split()))

    # Convert P to 0-based indexing
    # P[i] = j means element at original index j moves to index i.
    # The operation is A_new[i] = A_old[P[i]] (using 0-based indices).
    P = [p - 1 for p in P_1based]
    A0 = A_1based

    # 1. Find the P-cycle containing index 0 and calculate depths.
    # The cycle path from 0 under P: 0 -> P[0] -> P[P[0]] -> ...
    # The index reached after p steps is P^p[0]. The depth of P^p[0] is p.
    cycle_0_path = []
    cycle_0_depth = {}
    current_idx = 0
    step = 0
    visited_in_cycle = set()

    while current_idx not in visited_in_cycle:
        visited_in_cycle.add(current_idx)
        cycle_0_path.append(current_idx)
        cycle_0_depth[current_idx] = step
        current_idx = P[current_idx]
        step += 1

    # Cycle length
    m0 = len(cycle_0_path)

    # 2. Find the minimum value in A0 at indices within this cycle (the set of indices in cycle_0_path).
    min_val_0 = float('inf')
    for idx in cycle_0_path:
        min_val_0 = min(min_val_0, A0[idx])

    # 3. Find all indices j* in this cycle where A0 has the value min_val_0.
    min_val_indices_in_cycle_0 = [idx for idx in cycle_0_path if A0[idx] == min_val_0]

    # 4. Candidate powers k are the depths of these indices j* in the P-cycle path from 0.
    # A^(k)[0] = A0[P^k[0]]. This is minimal when P^k[0] is one of min_val_indices_in_cycle_0.
    # If j* = P^p[0], then P^k[0] = j* implies k \equiv p \pmod{m0}. The smallest non-negative k is p.
    candidate_k = sorted([cycle_0_depth[idx] for idx in min_val_indices_in_cycle_0])

    # 5. Initialize current candidates. These are indices into the sorted candidate_k list.
    current_candidate_indices = list(range(len(candidate_k)))

    # 6. Precompute P_step[j][i] = P^{2^j}[i] using binary lifting
    # To compute P^k[i], we need powers P^{2^j} where 2^j <= k. Max k < N.
    # Max j is floor(log2(N-1)) for N > 1. Number of levels = floor(log2(N-1)) + 1.
    # For N=1, max k=0.
    max_log_N = 0
    if N > 1:
        max_log_N = int(math.floor(math.log2(N - 1))) + 1
    elif N == 1:
         max_log_N = 1 # Need P_step array of size 1 for j=0, although get_Pk_i(0) doesn't use it.

    P_step = [[0] * N for _ in range(max_log_N)]
    if N > 0: # Constraint 1 <= N
        P_step[0] = list(P) # P^1

        for j in range(1, max_log_N):
            for i in range(N):
                P_step[j][i] = P_step[j-1][P_step[j-1][i]]

    # Function to compute P^k[i] using binary exponentiation (P_step array)
    def get_Pk_i(k, i, P_step):
        if k == 0:
            return i
        if N == 0: return i # Should not happen based on constraints

        res_i = i
        temp_k = k
        # Iterate through bits of k
        for j in range(max_log_N):
            if (temp_k >> j) & 1:
                 # Apply P^{2^j}
                res_i = P_step[j][res_i]
            # Optimization: if the largest remaining power 2^j is > temp_k, we can stop early.
            # But iterating max_log_N times is simpler and complexity is fine.
            # The original temp_k is needed to check remaining bits.
            # Correct loop iterates max_log_N times, applying power 2^j if the j-th bit is set in k.
        return res_i

    # 7. Iterate through indices i (0 to N-1) to filter candidates
    # We maintain indices into the `candidate_k` list.
    
    for i in range(N):
        # If only one candidate remains, we've found the unique best k
        if len(current_candidate_indices) <= 1:
            break

        # Evaluate values A^{(k)}[i] for current candidate indices
        values_at_i = []
        for cand_idx in current_candidate_indices:
            k = candidate_k[cand_idx]
            target_idx = get_Pk_i(k, i, P_step)
            values_at_i.append(A0[target_idx])

        # Find minimum value among active candidates at index i
        min_val_i = min(values_at_i)

        # Filter candidate indices: keep only those that achieve the minimum value at index i
        next_candidate_indices = []
        for j, val in zip(current_candidate_indices, values_at_i):
            if val == min_val_i:
                next_candidate_indices.append(j)

        current_candidate_indices = next_candidate_indices

    # The best k is the first one remaining in the filtered list (index into candidate_k)
    best_k_final = candidate_k[current_candidate_indices[0]]

    # 8. Compute the final array A^(best_k)
    A_final = [0] * N
    for i in range(N):
        target_idx = get_Pk_i(best_k_final, i, P_step)
        A_final[i] = A0[target_idx]

    # 9. Print the result
    sys.stdout.write(' '.join(map(str, A_final)) + '
')

solve()