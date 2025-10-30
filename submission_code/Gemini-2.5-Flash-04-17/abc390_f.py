import sys
import bisect

# Increase recursion depth for potentially deep calls (though not expected here)
sys.setrecursionlimit(2000)

def count_intervals(a, b):
    """Counts pairs (L, R) with a <= L <= R <= b"""
    if a > b:
        return 0
    length = b - a + 1
    return length * (length + 1) // 2

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Precompute positions of each value (0-based indexing for A, 1-based for positions)
    pos = [[] for _ in range(N + 2)] # Use N+2 to potentially store N+1 value positions
    for i in range(N):
        pos[A[i]].append(i + 1) # Use 1-based indexing for positions

    # Precompute sorted indices for values > v and >= v for optimization
    # P_gt_v[v] stores sorted list of indices i such that A[i-1] > v
    # P_ge_v[v] stores sorted list of indices i such that A[i-1] >= v
    P_gt_v = [[] for _ in range(N + 2)]
    P_ge_v = [[] for _ in range(N + 2)]

    current_gt = [] # Indices > N
    current_ge = [] # Indices >= N+1

    for v in range(N, 0, -1):
        # P_gt_{v-1} = pos[v] + P_gt_v
        # P_ge_{v-1} = pos[v] + P_ge_v (incorrect, should be P_ge_{v} + pos[v])
        
        # P_gt_{v-1} contains indices of values in {v, v+1, ..., N}
        # This should be P_ge_{v-1}
        P_ge_v[v-1] = sorted(pos[v] + P_ge_v[v])
        
        # P_gt_v contains indices of values in {v+1, ..., N}
        P_gt_v[v] = P_ge_v[v]


    total_sum = N * (N + 1) // 2 # Contribution from the base block (value 1)

    # Contribution from gaps: sum over v from 1 to N-1
    for v in range(1, N):
        # Indices of v+1: pos[v+1]
        # Add boundaries 0 and N+1 to define intervals where v+1 is absent
        pos_v_plus_1_padded = [0] + pos[v + 1] + [N + 1]

        # Iterate over intervals [a, b] where v+1 does not appear
        for i in range(len(pos_v_plus_1_padded) - 1):
            a = pos_v_plus_1_padded[i] + 1
            b = pos_v_plus_1_padded[i + 1] - 1

            if a > b:
                continue

            # We need to count pairs (L, R) with a <= L <= R <= b such that:
            # 1. v is present in A[L-1...R-1] (i.e., L <= p <= R for some p in pos[v])
            # 2. max value in A[L-1...R-1] is > v (i.e., L <= p <= R for some p in P_gt_v[v])
            # (Condition v+1 not present is guaranteed by [a, b])

            # Total pairs in [a, b]
            total_pairs_in_ab = count_intervals(a, b)

            # Pairs where v is NOT present in [L, R] (L, R within [a, b])
            # Indices of v in [a, b]
            v_indices_in_ab = pos[v][bisect.bisect_left(pos[v], a):bisect.bisect_right(pos[v], b)]
            v_indices_in_ab_padded = [a - 1] + v_indices_in_ab + [b + 1]
            pairs_no_v = 0
            for j in range(len(v_indices_in_ab_padded) - 1):
                sub_a = v_indices_in_ab_padded[j] + 1
                sub_b = v_indices_in_ab_padded[j + 1] - 1
                pairs_no_v += count_intervals(sub_a, sub_b)

            # Pairs where max value is NOT > v (i.e., max value is <= v) in [L, R] (L, R within [a, b])
            # Indices of values > v in [a, b] are slice of P_gt_v[v]
            gt_v_indices_in_ab = P_gt_v[v][bisect.bisect_left(P_gt_v[v], a):bisect.bisect_right(P_gt_v[v], b)]
            gt_v_indices_in_ab_padded = [a - 1] + gt_v_indices_in_ab + [b + 1]
            pairs_max_le_v = 0
            for j in range(len(gt_v_indices_in_ab_padded) - 1):
                sub_a = gt_v_indices_in_ab_padded[j] + 1
                sub_b = gt_v_indices_in_ab_padded[j + 1] - 1
                pairs_max_le_v += count_intervals(sub_a, sub_b)

            # Pairs where v is NOT present AND max value is NOT > v (i.e., max <= v) in [L, R] (L, R within [a, b])
            # This means all values must be < v
            # Indices of values >= v in [a, b] are slice of P_ge_v[v]
            ge_v_indices_in_ab = P_ge_v[v][bisect.bisect_left(P_ge_v[v], a):bisect.bisect_right(P_ge_v[v], b)]
            ge_v_indices_in_ab_padded = [a - 1] + ge_v_indices_in_ab + [b + 1]
            pairs_all_lt_v = 0
            for j in range(len(ge_v_indices_in_ab_padded) - 1):
                sub_a = ge_v_indices_in_ab_padded[j] + 1
                sub_b = ge_v_indices_in_ab_padded[j + 1] - 1
                pairs_all_lt_v += count_intervals(sub_a, sub_b)

            # Using inclusion-exclusion principle:
            # Count(v present AND max > v)
            # = Total - Count(v not present) - Count(max <= v) + Count(v not present AND max <= v)
            # = Total - Count(v not present) - Count(max <= v) + Count(all < v)
            count_v_present_and_max_gt_v = total_pairs_in_ab - pairs_no_v - pairs_max_le_v + pairs_all_lt_v
            total_sum += count_v_present_and_max_gt_v

    print(total_sum)

solve()