import sys
from collections import defaultdict
import bisect

def count_pairs_in_range(I_start, I_end):
    # Counts (L, R) such that I_start <= L <= R <= I_end
    S = I_end - I_start + 1
    if S <= 0: return 0
    return S * (S + 1) // 2

def count_pairs_not_intersecting_pos_v_in_range(I_start, I_end, pos_v):
    # Counts (L, R) such that I_start <= L <= R <= I_end and [L,R] does not intersect pos_v

    total_not_intersecting = 0
    
    # Get indices of v in [I_start, I_end]
    it_start = bisect.bisect_left(pos_v, I_start)
    it_end = bisect.bisect_right(pos_v, I_end)
    
    if it_start == it_end:
        # v does not appear in A[I_start..I_end], all pairs don't intersect
        return count_pairs_in_range(I_start, I_end)
        
    # Indices of v in [I_start, I_end]: pos_v[it_start : it_end]
    
    # Gaps within [I_start, I_end] where v does not appear
    # Before first occurrence: [I_start, pos_v[it_start] - 1]
    gap_start = I_start
    gap_end = pos_v[it_start] - 1
    total_not_intersecting += count_pairs_in_range(gap_start, gap_end)
    
    # Between occurrences
    for k in range(it_start, it_end - 1):
        gap_start = pos_v[k] + 1
        gap_end = pos_v[k+1] - 1
        total_not_intersecting += count_pairs_in_range(gap_start, gap_end)

    # After last occurrence: [pos_v[it_end-1] + 1, I_end]
    gap_start = pos_v[it_end-1] + 1
    gap_end = I_end
    total_not_intersecting += count_pairs_in_range(gap_start, gap_end)

    return total_not_intersecting


def count_pairs_with_v_in_range(I_start, I_end, pos_v):
    # Counts (L, R) such that I_start <= L <= R <= I_end and v is in A[L..R]
    # This is total pairs in range minus pairs not intersecting pos_v
    total_in_range = count_pairs_in_range(I_start, I_end)
    not_intersecting = count_pairs_not_intersecting_pos_v_in_range(I_start, I_end, pos_v)
    return total_in_range - not_intersecting


def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Store indices for each value (1-based)
    pos = defaultdict(list)
    for i, val in enumerate(A, 1):
        pos[val].append(i)

    total_sum = 0

    # Contribution from v = 1
    v = 1
    if v in pos:
        # Count pairs (L, R) such that 1 <= L <= R <= N and 1 is in A[L..R]
        total_sum += count_pairs_with_v_in_range(1, N, pos[v])

    # Contribution from v > 1
    for v in range(2, N + 1):
        # Count pairs (L, R) such that v is in A[L..R] AND v-1 is NOT in A[L..R]
        # v-1 is NOT in A[L..R] iff [L, R] is fully contained in a gap of pos[v-1]

        if v not in pos:
             continue # v never appears

        pos_prev = [0] + pos[v-1] + [N+1] # Add sentinels for gaps of v-1

        # Iterate over gaps of v-1
        for k in range(len(pos_prev) - 1):
            j_r = pos_prev[k]
            j_r_plus_1 = pos_prev[k+1]

            # The gap is indices (j_r, j_r_plus_1). Range is [j_r + 1, j_r_plus_1 - 1].
            I_start = j_r + 1
            I_end = j_r_plus_1 - 1

            if I_start > I_end:
                continue # Empty gap

            # Count pairs (L, R) such that I_start <= L <= R <= I_end and v is in A[L..R]
            # These are exactly the pairs where v-1 is NOT in A[L..R] AND v IS in A[L..R] (within this gap).
            total_sum += count_pairs_with_v_in_range(I_start, I_end, pos[v])

    print(total_sum)

solve()