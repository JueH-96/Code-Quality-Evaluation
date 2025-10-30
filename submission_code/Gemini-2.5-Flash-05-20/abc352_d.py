import collections
import sys

def solve():
    # Read N and K from standard input
    N, K = map(int, sys.stdin.readline().split())
    # Read permutation P from standard input
    # P is 1-indexed in problem statement, but read into a 0-indexed list
    P = list(map(int, sys.stdin.readline().split()))

    # Create a mapping from value to its 0-based index in P.
    # pos[v-1] stores the 0-based index `i` such that P[i] = v.
    # For example, if P = [2, 3, 1, 4] (P[0]=2, P[1]=3, P[2]=1, P[3]=4):
    # - Value 1 is at P[2], so pos[0] = 2
    # - Value 2 is at P[0], so pos[1] = 0
    # - Value 3 is at P[1], so pos[2] = 1
    # - Value 4 is at P[3], so pos[3] = 3
    # Resulting `pos` array: [2, 0, 1, 3]
    pos = [0] * N
    for i in range(N):
        pos[P[i]-1] = i 

    # Deques to maintain minimum and maximum indices within the current window of `pos` array.
    # These deques store indices `j` (from 0 to N-1) such that `pos[j]` is a candidate for min/max.
    # min_deque: stores indices `j` such that `pos[j]` values are in increasing order.
    # max_deque: stores indices `j` such that `pos[j]` values are in decreasing order.
    min_deque = collections.deque() 
    max_deque = collections.deque() 

    # Initialize the minimum span found so far to infinity.
    min_overall_span = float('inf')

    # Phase 1: Initialize deques with the first K elements.
    # These elements correspond to the `pos` values for consecutive integers from 1 to K.
    # The indices in the `pos` array are `0, 1, ..., K-1`.
    for i in range(K):
        # Add `pos[i]` to `min_deque`: remove elements from back if new element is smaller or equal.
        while min_deque and pos[min_deque[-1]] >= pos[i]:
            min_deque.pop()
        min_deque.append(i)

        # Add `pos[i]` to `max_deque`: remove elements from back if new element is larger or equal.
        while max_deque and pos[max_deque[-1]] <= pos[i]:
            max_deque.pop()
        max_deque.append(i)
    
    # After processing the first K elements, calculate the span for this initial window.
    # `pos[min_deque[0]]` gives the actual minimum 0-based index in P for values 1 to K.
    # `pos[max_deque[0]]` gives the actual maximum 0-based index in P for values 1 to K.
    current_min_idx_in_P = pos[min_deque[0]]
    current_max_idx_in_P = pos[max_deque[0]]
    min_overall_span = min(min_overall_span, current_max_idx_in_P - current_min_idx_in_P)

    # Phase 2: Slide the window.
    # `i` represents the 0-based index in the `pos` array of the new element entering the window.
    # This loop iterates `i` from `K` up to `N-1`.
    # For each `i`, the window in the `pos` array is `pos[i-K+1 : i+1]`.
    # This corresponds to consecutive value blocks like (2, ..., K+1), (3, ..., K+2), etc.
    for i in range(K, N):
        # Remove elements from the front of deques if their indices are outside the current window.
        # The leftmost index of the current window in the `pos` array is `i - K + 1`.
        # Any index `j` in the deque where `j < i - K + 1` should be removed.
        if min_deque and min_deque[0] < i - K + 1:
            min_deque.popleft()
        if max_deque and max_deque[0] < i - K + 1:
            max_deque.popleft()

        # Add the new element `pos[i]` to deques, maintaining their properties.
        # For `min_deque`:
        while min_deque and pos[min_deque[-1]] >= pos[i]:
            min_deque.pop()
        min_deque.append(i)

        # For `max_deque`:
        while max_deque and pos[max_deque[-1]] <= pos[i]:
            max_deque.pop()
        max_deque.append(i)

        # Calculate the span for the current window.
        # `min_deque[0]` and `max_deque[0]` give the indices within the `pos` array.
        # `pos[that_index]` then gives the actual 0-based index in the original `P` list.
        current_min_idx_in_P = pos[min_deque[0]]
        current_max_idx_in_P = pos[max_deque[0]]
        min_overall_span = min(min_overall_span, current_max_idx_in_P - current_min_idx_in_P)

    # Print the final result.
    sys.stdout.write(str(min_overall_span) + "
")

# Call the solve function to run the program.
solve()