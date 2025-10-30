import sys
from bisect import bisect_left, bisect_right

# Use sys.stdin.readline for faster input
input = sys.stdin.readline

def combinations2(n):
    """Calculates n C 2, the number of pairs (i, j) with 0 <= i < j < n.
    Equivalently, counts pairs (L, R) with a < L <= R < b where n = b-a."""
    if n < 2:
        return 0
    return n * (n - 1) // 2

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    # pos[v] stores 1-based indices where value v appears, plus sentinels 0 and N+1
    pos = [[] for _ in range(N + 1)]
    for i in range(N):
        pos[A[i]].append(i + 1)

    for v in range(1, N + 1):
        pos[v] = [0] + pos[v] + [N + 1]

    total_f_sum = 0
    
    # f(L,R) = |{s in D(L,R) | s-1 not in D(L,R)}| for non-empty D(L,R).
    # D(L,R) is non-empty for 1 <= L <= R <= N due to constraints 1 <= A_i <= N.
    # Total sum = sum over s=1 to N of count of (L,R) where s in D(L,R) and s-1 not in D(L,R)

    # Contribution for s=1: count of (L,R) where 1 in D(L,R) (since 0 is never in D(L,R))
    # Count = total number of pairs (L,R) - pairs where 1 is NOT in D(L,R)
    # Total pairs (L,R) with 1 <= L <= R <= N is combinations2(N+1) using boundary indices 0 and N+1.
    
    total_pairs_1_to_N = combinations2(N + 1)

    if len(pos[1]) > 2: # If 1 appears in A
        # Pairs (L,R) where 1 is NOT in D(L,R) are those fully contained in (pos[1][i], pos[1][i+1])
        pairs_not_containing_1 = 0
        for i in range(len(pos[1]) - 1):
            interval_length = pos[1][i+1] - pos[1][i]
            pairs_not_containing_1 += combinations2(interval_length)
        # Contribution for s=1 is the count of pairs containing 1
        total_f_sum += total_pairs_1_to_N - pairs_not_containing_1
    # Else (1 does not appear), count of (L,R) where 1 in D is 0. Contribution 0.

    # Contribution for s = 2, ..., N: count of (L,R) where s in D(L,R) and s-1 not in D(L,R)
    for s in range(2, N + 1):
        # If s does not appear, s in D(L,R) is never true. Count is 0.
        if len(pos[s]) <= 2:
            continue

        # Iterate through intervals (u_j, u_{j+1}) defined by indices of s-1.
        # Any pair (L,R) with u_j < L <= R < u_j_plus_1 satisfies s-1 not in D(L,R).
        # pos[s-1] will be [0, N+1] if s-1 does not appear, correctly giving one interval (0, N+1).
        P_s_minus_1 = pos[s-1]
        P_s = pos[s]

        for j in range(len(P_s_minus_1) - 1):
            u_j = P_s_minus_1[j]
            u_j_plus_1 = P_s_minus_1[j+1]

            # Total pairs (L,R) with u_j < L <= R < u_j_plus_1
            total_pairs_in_interval = combinations2(u_j_plus_1 - u_j)

            # Among these pairs, count those where s is NOT in D(L,R).
            # These are pairs (L,R) fully contained in (u_j, u_j_plus_1) AND containing no index of s.
            # Find indices of s that fall within (u_j, u_j_plus_1).
            # bisect_right finds first element > u_j.
            start_idx_s_in_interval = bisect_right(P_s, u_j)
            # bisect_left finds first element >= u_j_plus_1. Elements < u_j_plus_1 are before this index.
            end_idx_s_in_interval = bisect_left(P_s, u_j_plus_1)
            
            indices_s_in_interval = P_s[start_idx_s_in_interval : end_idx_s_in_interval]

            # These indices split (u_j, u_j_plus_1) into sub-intervals that contain no s.
            # Boundary indices of these sub-intervals are u_j, indices_s_in_interval, u_j_plus_1.
            w_indices = [u_j] + indices_s_in_interval + [u_j_plus_1]

            # Count pairs (L,R) within (u_j, u_j_plus_1) that do NOT contain s.
            # These are pairs fully contained in one of the sub-intervals (w_k, w_{k+1}).
            pairs_not_containing_s_in_interval = 0
            for k in range(len(w_indices) - 1):
                sub_interval_length = w_indices[k+1] - w_indices[k]
                pairs_not_containing_s_in_interval += combinations2(sub_interval_length)

            # Pairs (L,R) within (u_j, u_j_plus_1) that DO contain s contribute to the sum.
            # These pairs satisfy (s-1 not in D) AND (s in D).
            total_f_sum += total_pairs_in_interval - pairs_not_containing_s_in_interval
            
    print(total_f_sum)

solve()