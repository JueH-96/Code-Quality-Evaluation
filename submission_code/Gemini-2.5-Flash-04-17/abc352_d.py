import sys
from sortedcontainers import SortedList

def solve():
    # Read N and K
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    K = int(line1[1])

    # Read the permutation P (1-based values, read into 0-based list)
    P = list(map(int, sys.stdin.readline().split()))

    # Create a position array: pos[v] stores the 0-based index i such that P[i] = v
    # Values are 1 to N, so array size is N+1.
    pos = [0] * (N + 1)
    for i in range(N):
        pos[P[i]] = i

    # Use SortedList to efficiently maintain the sorted indices of values
    # in the current window {a, a+1, ..., a+K-1} and query min/max.
    sl = SortedList()

    # Initialize with the first window of values: {1, 2, ..., K}
    # These values have indices pos[1], pos[2], ..., pos[K].
    for v in range(1, K + 1):
        sl.add(pos[v])

    # The difference i_K - i_1 for the first window (a=1)
    min_diff = sl[-1] - sl[0]

    # Slide the window of values.
    # The window consists of K consecutive integers {a, a+1, ..., a+K-1}.
    # The possible values for 'a' are 1, 2, ..., N-K+1.
    # The initial step processed the window for a=1.
    # Now, we iterate through the transitions from window {a, ..., a+K-1} to {a+1, ..., a+K}.
    # This transition happens for 'a' values from 1 up to N-K.
    # In the loop where 'a' is the loop variable, it represents the value being removed.
    for a in range(1, N - K + 1):
        # Remove the index corresponding to the value leaving the window ('a')
        sl.remove(pos[a])
        # Add the index corresponding to the value entering the window ('a + K')
        sl.add(pos[a + K])

        # The sorted list now contains indices for values {a+1, ..., a+K}.
        # This corresponds to the window starting with value a+1.
        # Calculate the difference i_K - i_1 for this current window
        current_diff = sl[-1] - sl[0]

        # Update the minimum difference found so far
        min_diff = min(min_diff, current_diff)

    # Print the minimum difference
    print(min_diff)

# Ensure the solve function is called when the script is executed directly
if __name__ == "__main__":
    solve()