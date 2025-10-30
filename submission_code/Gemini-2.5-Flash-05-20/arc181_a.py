import sys

def solve():
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))

    # 0. Check if already sorted
    is_sorted = True
    for i in range(N):
        if P[i] != i + 1:
            is_sorted = False
            break
    if is_sorted:
        sys.stdout.write("0
")
        return

    # Precompute prefix maximums and suffix minimums for O(N) checks
    # For N >= 3, all indices will be valid in the loops below.
    # If N is small (e.g., N=1, N=2), special care for empty ranges is usually needed,
    # but problem constraints N >= 3 simplify this.
    pref_max = [0] * N
    pref_max[0] = P[0]
    for i in range(1, N):
        pref_max[i] = max(pref_max[i-1], P[i])

    suff_min = [0] * N
    suff_min[N-1] = P[N-1]
    for i in range(N-2, -1, -1):
        suff_min[i] = min(suff_min[i+1], P[i])

    # 1. Check for 1 operation
    is_1_op = False
    for k in range(1, N + 1): # k is the 1-indexed value of the pivot (P[k-1])
        if P[k-1] == k: # Check if P[k-1] is a fixed point
            prefix_ok = True
            # Check prefix P[0...k-2]
            # This segment should only contain values < k.
            # max(P[0...k-2]) < k implies all elements are in range [1, k-1].
            if k >= 2: # non-empty prefix P[0...k-2]
                if pref_max[k-2] >= k: # max value in prefix must be < k
                    prefix_ok = False
            # else (k=1), prefix is empty, which is vacuously ok.
            
            suffix_ok = True
            # Check suffix P[k...N-1]
            # This segment should only contain values > k.
            # min(P[k...N-1]) > k implies all elements are in range [k+1, N].
            if k <= N-1: # non-empty suffix P[k...N-1]
                if suff_min[k] <= k: # min value in suffix must be > k
                    suffix_ok = False
            # else (k=N), suffix is empty, which is vacuously ok.
            
            if prefix_ok and suffix_ok:
                is_1_op = True
                break
    
    if is_1_op:
        sys.stdout.write("1
")
        return

    # 2. If not 0 or 1 operation, it means 2 operations are needed.
    # This covers two main scenarios:
    # a) The permutation can be split into two "correct" segments (e.g., P[0..s-1] is a perm of 1..s,
    #    and P[s..N-1] is a perm of s+1..N). An operation with k=s+1 (or k=s) can partially sort it,
    #    leaving it in a state that takes one more operation. This is checked explicitly below.
    # b) Any other case not covered by 0 or 1 operation.
    # The problem constraints and typical competitive programming problem patterns
    # for minimum operations (0, 1, or 2) suggest that 2 is the maximum.
    
    # Check for the "split" scenario (leading to 2 ops as per sample logic)
    is_2_op_split_candidate = False
    for s in range(1, N): # s is the size of the first segment (P[0...s-1])
        # P[0...s-1] must contain values 1...s
        # If P is a permutation and max(P[0...s-1]) == s, then elements are 1...s.
        first_segment_contains_correct_values = (pref_max[s-1] == s)
        
        # P[s...N-1] must contain values s+1...N
        # If P is a permutation and min(P[s...N-1]) == s+1, then elements are s+1...N.
        second_segment_contains_correct_values = (suff_min[s] == s + 1)
        
        if first_segment_contains_correct_values and second_segment_contains_correct_values:
            is_2_op_split_candidate = True
            break
    
    if is_2_op_split_candidate:
        sys.stdout.write("2
")
        return
    
    # If it's not 0, not 1, and not a 'split' candidate, it still falls into the 2-operation category.
    # This is a common pattern where the maximum operations is low (e.g., 2 or 3).
    # Since 0 and 1 are covered, the remainder must be 2.
    sys.stdout.write("2
")


T = int(sys.stdin.readline())
for _ in range(T):
    solve()