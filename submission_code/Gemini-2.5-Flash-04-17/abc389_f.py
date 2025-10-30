import sys

# Fenwick tree for delta_D[1...size].
# D[j] = 1 + delta_D[j].
# Stores delta_D[j] at tree index j.
class FenwickTreeDeltaD:
    def __init__(self, size):
        # size = MAX_INIT_RATING - 1
        self.size = size # Number of elements in delta_D array (indices 1 to size)
        self.tree = [0] * (size + 1) # 1-based indexing

    def add(self, index, value):
        # Add value to delta_D[index] (1-based index)
        # Corresponds to tree index index
        i = index
        while i <= self.size:
            self.tree[i] += value
            i += i & (-i)

    def query_sum_delta_D(self, index):
        # Get sum delta_D[1]...delta_D[index] (1-based index)
        # Corresponds to tree sum up to index
        i = index
        s = 0
        # Ensure index is within tree bounds and non-negative for loop
        i = min(i, self.size)
        if i <= 0: return 0 # Sum delta_D[1]...delta_D[0] is empty sum

        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

# Binary search function to find the smallest initial rating v (1-based)
# such that its current rating >= target_val.
# The current rating of initial v is calculated based on the rating of initial 1
# after the previous contests and the accumulated delta_D from the BIT.
# rating_after_prev_contests[v] = rating_after_prev_contests[1] + (v-1) + sum_{j=1}^{v-1} delta_D[j] from BIT
# `current_rating_of_1` is the rating of initial 1 after the contests processed so far (before applying the current contest).
def lower_bound_initial_rating(max_initial_rating, target_val, bit_delta_D, current_rating_of_1):
    low = 1
    high = max_initial_rating + 1 # Search space [low, high)

    while low < high:
        mid = low + (high - low) // 2
        
        # Calculate current rating for initial rating 'mid'
        # rating[mid] = current_rating_of_1 + (mid - 1) + sum_{j=1}^{mid-1} delta_D[j]
        current_rating_at_mid = current_rating_of_1 + (mid - 1) + bit_delta_D.query_sum_delta_D(mid - 1)

        if current_rating_at_mid >= target_val:
            high = mid
        else:
            low = mid + 1
            
    return low # low is the smallest initial rating >= 1

# Main logic
def solve():
    # Use faster I/O
    input = sys.stdin.readline

    N = int(input())
    contests = []
    for _ in range(N):
        L, R = map(int, input().split())
        contests.append((L, R))

    Q = int(input())
    queries = []
    for _ in range(Q):
        queries.append(int(input()))

    MAX_INIT_RATING = 500000
    
    # BIT to store the accumulated delta_D[v] = (rating_k[v+1] - rating_k[v]) - (rating_{k-1}[v+1] - rating_{k-1}[v])
    # over all contests k processed so far.
    # It stores the total deviation from the base difference of 1.
    # BIT size is MAX_INIT_RATING - 1. Indices 1 to MAX_INIT_RATING - 1.
    # delta_D_total[v] is stored at BIT index v.
    # The BIT tracks differences for initial ratings 1 to MAX_INIT_RATING.
    # D[v] = rating[v+1] - rating[v] for v = 1...MAX_INIT_RATING-1.
    # We store delta_D[v] = D[v] - 1 at BIT index v.
    bit_delta_D = FenwickTreeDeltaD(MAX_INIT_RATING - 1)

    # Track the current rating of initial rating 1 separately
    current_rating_of_1 = 1

    # Process each contest
    for L, R in contests:
        # Before applying contest (L, R), current_rating_of_1 holds the rating after the previous contest.
        # We use this value in binary search to find the boundary initial ratings based on ratings *before* the current contest.

        # Find v_start = smallest initial rating v (1-based, in [1, MAX_INIT_RATING])
        # such that its current rating (before this contest) >= L
        v_start = lower_bound_initial_rating(MAX_INIT_RATING, L, bit_delta_D, current_rating_of_1)

        # Find v_end_plus_1 = smallest initial rating v (1-based, in [1, MAX_INIT_RATING])
        # such that its current rating (before this contest) >= R + 1
        v_end_plus_1 = lower_bound_initial_rating(MAX_INIT_RATING, R + 1, bit_delta_D, current_rating_of_1)

        # The initial ratings v in the range [v_start, v_end_plus_1 - 1] get their rating increased by 1 in this contest.
        # This means the total increase for v in this range goes up by 1.
        # This changes the total delta_D[j] for j = v_start - 1 and j = v_end_plus_1 - 1.

        # Update total delta_D at index v_start - 1 by +1.
        # This corresponds to BIT index v_start - 1. Valid if v_start - 1 >= 1, i.e., v_start >= 2.
        if v_start >= 2:
             bit_delta_D.add(v_start - 1, 1)

        # Update total delta_D at index v_end_plus_1 - 1 by -1.
        # BIT index v_end_plus_1 - 1.
        # Valid if v_end_plus_1 - 1 >= 1 and v_end_plus_1 - 1 <= MAX_INIT_RATING - 1.
        # v_end_plus_1 - 1 >= 1 means v_end_plus_1 >= 2.
        # v_end_plus_1 - 1 <= MAX_INIT_RATING - 1 means v_end_plus_1 <= MAX_INIT_RATING.
        # So update if v_end_plus_1 is in [2, MAX_INIT_RATING].
        if v_end_plus_1 >= 2 and v_end_plus_1 <= MAX_INIT_RATING:
             bit_delta_D.add(v_end_plus_1 - 1, -1)

        # Now, apply the current contest (L, R) to current_rating_of_1 to get its value *after* this contest.
        if L <= current_rating_of_1 <= R:
            current_rating_of_1 += 1


    # After all contests, compute final ratings for queries
    results = []
    for X in queries:
        # Final rating for initial rating X is rating_N[X].
        # rating_N[X] = rating_N[1] + (X - 1) + sum_{j=1}^{X-1} delta_D_total[j]
        # rating_N[1] is current_rating_of_1 after all contests (which is its final value).
        # sum_{j=1}^{X-1} delta_D_total[j] = bit_delta_D.query_sum_delta_D(X - 1).
        
        # Formula: final_rating = final_rating_of_1 + (X - 1) + bit_delta_D.query_sum_delta_D(X - 1)
        # This formula holds for X >= 1 because query_sum_delta_D(0) returns 0.
        
        final_rating = current_rating_of_1 + (X - 1) + bit_delta_D.query_sum_delta_D(X - 1)
        results.append(final_rating)

    for res in results:
        print(res)

solve()