import sys
from bisect import bisect_left

# Use 0-indexed arrays internally
# Pieces A: A[0...N-1]
# Doubled array B: B[0...2N-1] where B[i] = A[i % N]
# Prefix sums Q: Q[i] = sum(B[0...i-1]), Q[0]=0. Size 2N+1. Q[i] is sum of first i pieces.

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    B = A + A
    Q = [0] * (2 * N + 1)
    for i in range(2 * N):
        Q[i+1] = Q[i] + B[i]

    # Jump[i][p] = min end index of a segment starting at piece i, after 2^p segments >= X.
    # Index i refers to the starting piece index in B (0 to 2N-1).
    LOGK = (K + 1).bit_length() # Max steps K

    def build_jump_table(X):
        # Jump[i][0] = min end index j >= i such that sum(B[i...j]) >= X.
        # Sum B[i...j] is Q[j+1] - Q[i]. We need Q[j+1] >= Q[i] + X.
        # Search k=j+1 in Q, k >= i+1.
        Jump = [[2*N] * LOGK for _ in range(2*N)]

        for i in range(2*N):
             target_q = Q[i] + X
             # bisect_left searches in Q[lo:hi]
             k = bisect_left(Q, target_q, lo=i+1, hi=2*N+1)
             if k < 2*N+1:
                 Jump[i][0] = k - 1 # end index in B

        for p in range(1, LOGK):
            for i in range(2*N):
                if Jump[i][p-1] < 2*N:
                    next_start_piece = Jump[i][p-1] + 1
                    if next_start_piece < 2*N:
                         Jump[i][p] = Jump[next_start_piece][p-1]

        return Jump

    # Get end index after `steps` segments starting at `start_idx` in B.
    def get_jump(Jump, start_idx, steps):
        curr = start_idx
        if curr >= 2*N: return 2*N
        for p in range(LOGK):
            if (steps >> p) & 1:
                curr = Jump[curr][p]
                if curr >= 2*N: return 2*N
        return curr

    # M(s, X) = min segments to partition B[s...s+N-1] greedily starting at s.
    # Find min M such that segment M starting at s ends at or after s+N-1.
    def count_min_segments(start_piece_idx, total_len, Jump):
        end_idx = start_piece_idx + total_len - 1
        
        # Binary search for steps M in [1, N+1]
        low_m, high_m = 1, N + 2
        result_m = float('inf')
        
        while low_m < high_m:
            mid_m = (low_m + high_m) // 2
            final_pos = get_jump(Jump, start_piece_idx, mid_m)
            if final_pos >= end_idx:
                result_m = mid_m
                high_m = mid_m
            else:
                low_m = mid_m + 1
        return result_m

    # Binary search for x
    low = 1
    high = sum(A) + 1
    x = 0

    while low < high:
        mid = (low + high) // 2
        Jump_mid = build_jump_table(mid)
        
        can = False
        for s in range(N):
             min_segs = count_min_segments(s, N, Jump_mid)
             if min_segs <= K:
                 can = True
                 break
        
        if can:
            x = mid
            low = mid + 1
        else:
            high = mid

    # Now find the number of never cut lines.
    # Cut line i (between piece i and i+1) is potentially cut if there exists an optimal
    # partition where a segment ends at piece i.
    # Piece i corresponds to B[i] and B[i+N]. So a segment ends at index i or i+N in B.
    # An optimal partition starting at s uses segments $B_s..e_1, B_{e_1+1}..e_2, \ldots, B_{e_{K-1}+1}..e_K=s+N-1$,
    # where sums >= x. The cut lines are $e_1 \pmod N, e_2 \pmod N, \ldots, e_{K-1} \pmod N$.
    
    Jump_x = build_jump_table(x)

    # Set of cut lines that can be cut
    can_be_cut = [False] * N

    for s in range(N): # Iterate through all possible start pieces B[s]
        # If the greedy partition from s uses <= K segments and covers N pieces,
        # it is one valid optimal partition. The cuts are after indices get_jump(s, j) for j=1..M(s,x)-1.
        
        min_segs_s = count_min_segments(s, N, Jump_x)
        
        # If M(s,x) <= K, then partitioning starting at s into K segments is possible.
        # We need to find all possible end indices e_j for j = 1..K-1.
        # e_j is end of j-th segment starting at s.
        # e_j must be in range [s+j-1, s+N-K+j-1].
        # Also e_j >= get_jump(s, j).
        # Also, s+N-1 must be reachable from e_j+1 in K-j segments: get_jump(e_j+1, K-j) <= s+N-1.
        
        # Find max start e_prime such that get_jump(e_prime, k_steps) <= target_end
        def find_max_reachable_start(k_steps, target_end):
             low_e, high_e = 0, target_end + 1
             ans_e = -1 # Initialize with an invalid index

             while low_e < high_e:
                 mid_e = (low_e + high_e) // 2
                 if mid_e >= 2*N: # Index out of bounds for B/Jump
                      high_e = mid_e
                      continue

                 if get_jump(Jump_x, mid_e, k_steps) <= target_end:
                     ans_e = mid_e
                     low_e = mid_e + 1
                 else:
                     high_e = mid_e
             return ans_e

        for j in range(1, K): # j is the segment number, end index is e_j
             # Possible range for e_j is [s+j-1, s+N-K+j-1].
             range_low = s + j - 1
             range_high = s + N - K + j - 1

             # Lower bound from reachability from start s in j steps
             L_sj = get_jump(Jump_x, s, j)
             
             # Upper bound from reachability to end s+N-1 from e_j+1 in K-j segments
             # Need get_jump(e_j+1, K-j) <= s+N-1.
             # Max e_j+1 such that get_jump(e_j+1, K-j) <= s+N-1. Let this be R_plus_one_sj.
             # We search for e_j+1, so the potential indices are >= s+j. Max possible end is s+N-1, so max start is s+N-1 - (K-j) + 1 = s+N-K+j.
             
             # Indices we search for e_j+1 are in range [s+j, s+N-K+j].
             # If K-j=0, jump is 0 steps, end is start. get_jump(e_j+1, 0) = e_j+1.
             # So e_j+1 <= s+N-1 => e_j <= s+N-2.
             # If j=K-1, K-j=1. jump 1 step from e_j+1.
             # Range for e_j is [s+K-2, s+N-2]. e_j+1 is [s+K-1, s+N-1].
             # R_plus_one_sj is max index i in B such that get_jump(i, K-j) <= s+N-1.
             # Search space for i is [0, 2N-1].
             R_plus_one_sj = find_max_reachable_start(K-j, s+N-1)

             R_sj = R_plus_one_sj - 1 # Corresponding max e_j

             # Possible endpoints for e_j are in [max(range_low, L_sj), min(range_high, R_sj)]
             # Also must ensure that the number of segments required from s to high_ej is <= j
             
             low_ej = max(range_low, L_sj)
             high_ej = min(range_high, R_sj)

             # If low_ej <= high_ej, then any index e in [low_ej, high_ej] can be a possible e_j.
             # The cut line index is e mod N.
             if low_ej <= high_ej:
                 start_mod = low_ej % N
                 end_mod = high_ej % N
                 
                 if start_mod <= end_mod:
                     for idx in range(start_mod, end_mod + 1):
                         can_be_cut[idx] = True
                 else: # Wraps around
                     for idx in range(start_mod, N):
                         can_be_cut[idx] = True
                     for idx in range(0, end_mod + 1):
                         can_be_cut[idx] = True

    y = N - sum(can_be_cut)

    print(x, y)

solve()