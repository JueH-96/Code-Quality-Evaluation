# YOUR CODE HERE
import heapq
import sys

# Faster input reading
input = sys.stdin.readline

# Read input
N, M, L = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Store prohibited pairs in a set for efficient lookup (using 0-based indexing)
prohibited_pairs = set()
for _ in range(L):
    c, d = map(int, input().split())
    prohibited_pairs.add((c - 1, d - 1))

# Create list of (cost, original_index) for main dishes and sort descending
# sa[i] = (cost_of_i_th_most_expensive_main_dish, original_index_of_that_dish)
sa = [(a[i], i) for i in range(N)]
sa.sort(key=lambda x: x[0], reverse=True)

# Create list of (cost, original_index) for side dishes and sort descending
# sb[j] = (cost_of_j_th_most_expensive_side_dish, original_index_of_that_dish)
sb = [(b[j], j) for j in range(M)]
sb.sort(key=lambda x: x[0], reverse=True)

# Priority queue (min-heap) storing (-price, sa_idx, sb_idx)
# sa_idx is the index in the sorted main dishes list `sa`
# sb_idx is the index in the sorted side dishes list `sb`
pq = []
# Keep track of visited (sa_idx, sb_idx) pairs to avoid duplicates
# This set stores pairs of indices *within the sorted lists* `sa` and `sb` that have been added to the PQ
visited_indices = set()

# Add the most expensive possible combination to the PQ
# This corresponds to the first element in sorted 'sa' (index 0) and first in sorted 'sb' (index 0)
initial_sa_idx = 0
initial_sb_idx = 0
initial_price = sa[initial_sa_idx][0] + sb[initial_sb_idx][0]

heapq.heappush(pq, (-initial_price, initial_sa_idx, initial_sb_idx))
visited_indices.add((initial_sa_idx, initial_sb_idx))

# Explore until we find an allowed pair
while pq:
    # Get the pair with the maximum price (minimum negative price)
    current_neg_price, i_sa, j_sb = heapq.heappop(pq)
    current_price = -current_neg_price

    # Get the original indices of the main and side dishes using the indices in the sorted lists
    original_a_idx = sa[i_sa][1]
    original_b_idx = sb[j_sb][1]

    # Check if this original pair (main_dish_idx, side_dish_idx) is prohibited
    if (original_a_idx, original_b_idx) not in prohibited_pairs:
        # This is the first pair popped from the PQ that is not prohibited.
        # Since PQ extracts in decreasing order of price, this is the most expensive offered meal.
        print(current_price)
        break # Found the answer

    # If the current pair was prohibited, add the next possible expensive combinations to the PQ.
    # These combinations involve taking the current item from one sorted list and the next item from the other.

    # 1. Consider the main dish sa[i_sa] and the next side dish in the sorted list sb[j_sb+1]
    # The indices in the sorted lists would be (i_sa, j_sb + 1).
    next_sa_idx_1 = i_sa
    next_sb_idx_1 = j_sb + 1
    
    # Check if the next side dish index in the sorted list is valid (within bounds of sb)
    if next_sb_idx_1 < M:
        # Check if this combination of sorted indices (sa_idx, sb_idx) has already been added to the PQ
        if (next_sa_idx_1, next_sb_idx_1) not in visited_indices:
            # Calculate the price for this potential combination
            next_price_1 = sa[next_sa_idx_1][0] + sb[next_sb_idx_1][0]
            # Add to the PQ (using negative price for min-heap)
            heapq.heappush(pq, (-next_price_1, next_sa_idx_1, next_sb_idx_1))
            # Mark this combination of sorted indices as visited
            visited_indices.add((next_sa_idx_1, next_sb_idx_1))

    # 2. Consider the next main dish in the sorted list sa[i_sa+1] and the current side dish sb[j_sb]
    # The indices in the sorted lists would be (i_sa + 1, j_sb).
    next_sa_idx_2 = i_sa + 1
    next_sb_idx_2 = j_sb

    # Check if the next main dish index in the sorted list is valid (within bounds of sa)
    if next_sa_idx_2 < N:
        # Check if this combination of sorted indices (sa_idx, sb_idx) has already been added to the PQ
        if (next_sa_idx_2, next_sb_idx_2) not in visited_indices:
            # Calculate the price for this potential combination
            next_price_2 = sa[next_sa_idx_2][0] + sb[next_sb_idx_2][0]
            # Add to the PQ (using negative price for min-heap)
            heapq.heappush(pq, (-next_price_2, next_sa_idx_2, next_sb_idx_2))
            visited_indices.add((next_sa_idx_2, next_sb_idx_2))

# The loop is guaranteed to find an answer because L < NM, so at least one pair is allowed.