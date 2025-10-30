def solve():
    N, S = map(int, input().split())
    A = list(map(int, input().split()))
    
    total_sum = sum(A)
    
    # Compute prefix sums
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + A[i]
    
    # Case 1: subsequence entirely within one period
    for l in range(N):
        for r in range(l, N):
            if prefix_sum[r + 1] - prefix_sum[l] == S:
                return "Yes"
    
    # Case 2: subsequence spanning multiple periods
    for l_mod in range(N):  # starting position within period
        for r_mod in range(N):  # ending position within period
            suffix_sum = prefix_sum[N] - prefix_sum[l_mod]  # sum from l_mod to end of period
            prefix_sum_r = prefix_sum[r_mod + 1]  # sum from start of period to r_mod
            
            # We want: suffix_sum + k * total_sum + prefix_sum_r = S
            # So: k = (S - suffix_sum - prefix_sum_r) / total_sum
            
            target = S - suffix_sum - prefix_sum_r
            
            if target >= 0 and target % total_sum == 0:
                return "Yes"
    
    return "No"

print(solve())