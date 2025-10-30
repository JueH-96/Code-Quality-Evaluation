import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    distinct_values_list = sorted(list(set(A)))
    M = len(distinct_values_list)

    val_to_idx = {val: i for i, val in enumerate(distinct_values_list)}
    
    # Baseline answer: N deletion operations, 0 swaps.
    ans = N

    # Threshold for M. If M is larger, M^2 * 2^M DP is too slow.
    # Max M for M^2 * 2^M to pass is around 16-17.
    # M=17: 17^2 * 2^17 ~ 3.7 * 10^7. This might pass.
    M_THRESHOLD = 17 
    if M == 0: # Handles N=0 edge case if constraints allowed it (N>=2 here)
        sys.stdout.write("0
")
        return
    if M > M_THRESHOLD:
        sys.stdout.write(str(ans) + "
")
        return

    # Precompute C[j][k]: number of times value (mapped to) j appears BEFORE value (mapped to) k
    # Overall O(N*M) for this precomputation step.
    # Sum of N*M over test cases: M_THRESHOLD * sum(N) = 17 * 2e5 = 3.4e6, fine.
    C = [[0] * M for _ in range(M)]
    
    current_counts_for_C_table = [0] * M 
    for val_in_A in A:
        k_idx = val_to_idx[val_in_A] # current element's type index
        
        for j_idx in range(M):
            if j_idx == k_idx:
                continue
            # Add counts of type j_idx seen so far to C[j_idx][k_idx]
            # This means these j_idx's occurred before this instance of k_idx
            C[j_idx][k_idx] += current_counts_for_C_table[j_idx]
            
        current_counts_for_C_table[k_idx] += 1

    # DP state: dp[mask] = min cost to eliminate types in mask
    # Cost here is sum of (1 per deletion + swaps for that deletion)
    dp = [float('inf')] * (1 << M)
    dp[0] = 0 # Cost to eliminate no types is 0.

    for mask_old in range(1 << M): # Represents types already eliminated
        if dp[mask_old] == float('inf'):
            continue
        
        for x_idx in range(M): # Try to eliminate type x_idx next
            if not (mask_old & (1 << x_idx)):  # If x_idx is not yet eliminated in mask_old
                
                swaps_for_x = 0
                # Sum C[y_idx][x_idx] for all y_idx NOT in mask_old and y_idx != x_idx.
                # These are types y that are still present when x is chosen for elimination.
                for y_idx in range(M):
                    if y_idx == x_idx: 
                        continue
                    if not (mask_old & (1 << y_idx)): # If y_idx is still present
                        swaps_for_x += C[y_idx][x_idx]
                
                mask_new = mask_old | (1 << x_idx)
                cost_to_elim_x_this_step = 1 + swaps_for_x # 1 for deletion op
                
                if dp[mask_old] + cost_to_elim_x_this_step < dp[mask_new]:
                    dp[mask_new] = dp[mask_old] + cost_to_elim_x_this_step
                
    # dp[(1 << M) - 1] is the minimum cost using Strategy 2
    if M > 0 : # if there are any distinct values
         ans = min(ans, dp[(1 << M) - 1])
    elif N == 0 : # M=0 implies N=0
         ans = 0
    # else N > 0 but M=0, this case shouldn't happen if N>=2 implies elements.
    # If A is empty, N=0, M=0, ans=0.

    sys.stdout.write(str(ans) + "
")

num_test_cases = int(sys.stdin.readline())
for _ in range(num_test_cases):
    solve()