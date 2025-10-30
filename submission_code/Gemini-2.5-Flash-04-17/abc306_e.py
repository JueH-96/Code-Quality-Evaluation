import sys
import heapq

# Function to clean the top_k_heap (min-heap)
# Removes outdated elements from the top of the heap. An element (val, idx) is
# outdated if the current value in A[idx] is different from val, or if idx
# is no longer conceptually in the top K set according to is_in_top_k_dict.
def clean_top_k_heap(h, is_in_top_k_dict, A):
    while h:
        val, idx = h[0]
        # Check if this element is outdated or its index is no longer in the top K set
        if A[idx] != val or not is_in_top_k_dict.get(idx, False):
            heapq.heappop(h)
        else:
            break

# Function to clean the rest_heap (max-heap, stored as min-heap of negated values)
# Removes outdated elements from the top of the heap. An element (-val, idx) is
# outdated if the current value in A[idx] is different from val, or if idx
# is now conceptually in the top K set according to is_in_top_k_dict (should not be here).
def clean_rest_heap(h, is_in_top_k_dict, A):
    while h:
        neg_val, idx = h[0]
        val = -neg_val
        # Check if this element is outdated or its index is now in the top K set
        if A[idx] != val or is_in_top_k_dict.get(idx, False):
            heapq.heappop(h)
        else:
            break

# Function to peek the actual minimum value in top_k_heap after cleaning
# Returns float('inf') if the heap becomes empty after cleaning.
def peek_min_val(h):
    return h[0][0] if h else float('inf')

# Function to peek the actual maximum value in rest_heap after cleaning
# Returns float('-inf') if the heap becomes empty after cleaning.
def peek_max_val(h):
    return -h[0][0] if h else float('-inf')

# Read input
N, K, Q = map(int, sys.stdin.readline().split())

# Initialize array A (1-indexed) and tracking dict
A = [0] * (N + 1)
is_in_top_k_dict = {} # Map index (1 to N) to boolean: True if conceptually in top K set, False otherwise

# Initialize heaps and sum
top_k_heap = [] # min-heap of (value, index) pairs conceptually in the top K set
rest_heap = []  # min-heap of (-value, index) pairs conceptually in the rest set (simulates max-heap)
current_sum = 0

# Initial state: all elements are 0. The K largest are 0.
# Arbitrarily assign the first K indices to the conceptual top K set.
# Push all N initial elements (value 0) into the appropriate heap based on this assignment.
for i in range(1, N + 1):
    if i <= K:
        # Constraint 1 <= K <= N means K > 0 is always true.
        is_in_top_k_dict[i] = True
        heapq.heappush(top_k_heap, (0, i))
        current_sum += 0
    else:
        is_in_top_k_dict[i] = False
        heapq.heappush(rest_heap, (-0, i))

# Process queries
for _ in range(Q):
    X, Y = map(int, sys.stdin.readline().split())

    old_Y = A[X] # Get the old value at index X
    A[X] = Y     # Update the value in the array A

    # Check if index X was conceptually in the top K set before this update
    was_in_top_k = is_in_top_k_dict.get(X, False)

    # Step 1: Account for the removal of the old value at index X
    if was_in_top_k:
        # old_Y was conceptually in the top K set. Remove its contribution from sum.
        # Mark index X as conceptually leaving the top K set temporarily.
        is_in_top_k_dict[X] = False
        current_sum -= old_Y
        # The pair (old_Y, X) in top_k_heap is now outdated.

    # Step 2: Add the new value Y at index X to the set of elements being considered.
    # Add the new element (value Y, index X) to the rest heap first.
    # This simplifies the logic for Step 3.
    heapq.heappush(rest_heap, (-Y, X))
    # The index X is now conceptually not in the top K set (either it wasn't before, or we just marked it False).

    # Step 3: Rebalance count. Ensure exactly K elements are marked True in is_in_top_k_dict.
    # If was_in_top_k was True, the count of elements marked True just decreased by 1.
    # We need to move one element from the rest set to the top K set to maintain count K.
    if was_in_top_k:
        # Need to find the largest valid element in rest_heap and move it.
        clean_rest_heap(rest_heap, is_in_top_k_dict, A)
        # If rest_heap is not empty (i.e., N-K > 0), pop the max.
        # If K=N, rest_heap is empty, this branch implies was_in_top_k is true, which is always the case.
        # But if K=N, rest_heap is empty, so the `if rest_heap:` check handles it correctly.
        if rest_heap:
             neg_v_rest, i_rest = heapq.heappop(rest_heap)
             v_rest = -neg_v_rest # Get the actual value
             # Move (v_rest, i_rest) to top K set.
             is_in_top_k_dict[i_rest] = True # Mark this new index as in top K
             current_sum += v_rest
             heapq.heappush(top_k_heap, (v_rest, i_rest)) # Add to top_k_heap

    # After the above, the count of elements marked True in is_in_top_k_dict is now K
    # (unless N-K=0 and was_in_top_k was True, in which case it was N-1 but the rest_heap was empty,
    # and the general logic for K=N handles sum correctly even if counts are temporarily off).

    # Step 4: Ensure the value-based partition is correct.
    # The minimum value in the conceptual top K set must be >= the maximum value
    # in the conceptual rest set.
    # Swap elements if this property is violated.

    # Ensure heaps are clean at the top before checking swap condition
    clean_top_k_heap(top_k_heap, is_in_top_k_dict, A)
    clean_rest_heap(rest_heap, is_in_top_k_dict, A)

    # While the smallest in top_k is less than the largest in rest, swap them.
    # Condition: top_k_heap not empty (K > 0), rest_heap not empty (N-K > 0), AND min_top_k < max_rest.
    # The peek functions handle empty cases by returning +/- inf.
    while peek_min_val(top_k_heap) < peek_max_val(rest_heap):
        # Clean again before popping for swap, in case elements became outdated
        # while previous swap iterations were running.
        clean_top_k_heap(top_k_heap, is_in_top_k_dict, A)
        clean_rest_heap(rest_heap, is_in_top_k_dict, A)

        # Re-check condition after cleaning. Check if heaps became empty.
        if peek_min_val(top_k_heap) >= peek_max_val(rest_heap):
             break # Stop swapping if condition no longer met

        # Pop valid elements for swapping.
        # Guaranteed not empty by the loop condition and checks above.
        v_top, i_top = heapq.heappop(top_k_heap)
        neg_v_rest, i_rest = heapq.heappop(rest_heap)
        v_rest = -neg_v_rest

        # Swap their status in is_in_top_k_dict and update sum.
        # v_top moves from top_k to rest
        is_in_top_k_dict[i_top] = False
        current_sum -= v_top # Subtract from top_k sum
        heapq.heappush(rest_heap, (-v_top, i_top)) # Push to rest_heap (as negated value)

        # v_rest moves from rest to top_k
        is_in_top_k_dict[i_rest] = True
        current_sum += v_rest # Add to top_k sum
        heapq.heappush(top_k_heap, (v_rest, i_rest)) # Push to top_k_heap

    # After all rebalancing, current_sum is the sum of the K largest elements.
    print(current_sum)