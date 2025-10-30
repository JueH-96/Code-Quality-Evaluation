# YOUR CODE HERE
import sys
# Import deque from collections for efficient queue operations
from collections import deque

# Define the main function to solve the problem
def solve():
    # Read integers N and K from standard input
    # N: size of the permutation
    # K: length of the required subsequence
    N, K = map(int, sys.stdin.readline().split())
    
    # Read the permutation P as a list of integers
    P = list(map(int, sys.stdin.readline().split()))

    # Handle the base case where K=1. 
    # Any single element subsequence is a rearrangement of 1 consecutive integer.
    # The indices are (i_1). The difference i_K - i_1 = i_1 - i_1 = 0.
    # The minimum difference is always 0 for K=1.
    if K == 1:
        print(0)
        return

    # Create a position array 'pos' to store the 0-based index of each value in P.
    # pos[v] will store the index i such that P[i] = v.
    # The array size is N+1 to use 1-based indexing for values (1 to N). Index 0 remains unused.
    pos = [0] * (N + 1)
    for i in range(N):
        # P[i] is the value at 0-based index i. Store this index i in pos[P[i]].
        pos[P[i]] = i 

    # Initialize two deques for the sliding window minimum/maximum algorithm.
    # These deques will store tuples (position, value).
    # min_deque helps find the minimum position among elements in the current window.
    # max_deque helps find the maximum position among elements in the current window.
    min_deque = deque() 
    max_deque = deque() 
    
    # Initialize the minimum difference found so far.
    # The maximum possible difference between indices is N-1 (index N-1 minus index 0).
    # Initializing min_diff with N ensures any valid difference found will be smaller.
    min_diff = N 

    # Iterate through values x from 1 to N. This simulates a sliding window over the *values*.
    # The window at step x considers values {x-K+1, ..., x}.
    for x in range(1, N + 1):
        # Get the 0-based index (position) in the permutation P for the current value x.
        current_pos = pos[x]
        
        # Maintain the min_deque:
        # Remove elements from the right end (back) if their position is greater than or equal to current_pos.
        # This ensures that elements in the deque are sorted by position non-decreasingly,
        # and the element with the minimum position is always at the left end (front).
        while min_deque and min_deque[-1][0] >= current_pos:
            min_deque.pop()
        # Add the current element (position, value) to the right end of min_deque.
        min_deque.append((current_pos, x))

        # Maintain the max_deque:
        # Remove elements from the right end (back) if their position is less than or equal to current_pos.
        # This ensures that elements in the deque are sorted by position non-increasingly,
        # and the element with the maximum position is always at the left end (front).
        while max_deque and max_deque[-1][0] <= current_pos:
            max_deque.pop()
        # Add the current element (position, value) to the right end of max_deque.
        max_deque.append((current_pos, x))

        # Check if the window has reached its full size K. This happens when x >= K.
        # The window currently contains indices corresponding to values {x-K+1, ..., x}.
        if x >= K:
            # Remove elements from the left end (front) of the deques if they are no longer part of the current window.
            # An element (pos[x'], x') is outdated if its value x' is less than x-K+1,
            # which is equivalent to x' <= x-K.
            
            # Remove outdated elements from min_deque.
            while min_deque and min_deque[0][1] <= x - K:
                min_deque.popleft()
            
            # Remove outdated elements from max_deque.
            while max_deque and max_deque[0][1] <= x - K:
                max_deque.popleft()

            # After removing outdated elements, the front elements of the deques contain the relevant min/max positions.
            # Get the minimum position for the current window from the front of min_deque.
            current_min_pos = min_deque[0][0]
            # Get the maximum position for the current window from the front of max_deque.
            current_max_pos = max_deque[0][0]
            
            # Calculate the difference between the maximum and minimum positions for this window.
            current_diff = current_max_pos - current_min_pos
            
            # Update the overall minimum difference found across all windows considered so far.
            min_diff = min(min_diff, current_diff)

    # After iterating through all possible windows (corresponding to values 1..K up to N-K+1..N),
    # print the minimum difference found.
    print(min_diff)

# Execute the solve function to run the program logic.
solve()