# YOUR CODE HERE
import collections
import sys

def solve():
    """
    Solves the problem by reading from stdin and writing to stdout.
    """
    # Use fast I/O
    input = sys.stdin.readline

    # Read N and K from the first line of input
    try:
        line1 = input().split()
        if not line1: return
        N, K = int(line1[0]), int(line1[1])
    except (IOError, ValueError):
        return

    # Read the permutation P from the second line
    try:
        line2 = input().split()
        if not line2: return
        P = [int(x) for x in line2]
    except (IOError, ValueError):
        return

    # Handle the trivial case where K=1.
    # The difference i_K - i_1 will be i_1 - i_1 = 0.
    if K == 1:
        print(0)
        return

    # Precompute the 0-based positions of each value.
    # pos[v] will store the 0-based index of value v+1.
    pos = [0] * N
    for i in range(N):
        # P[i] is 1-based, so P[i]-1 is the 0-based value.
        pos[P[i] - 1] = i

    # Deques to store 0-indexed values `v` for finding min/max positions in the window.
    min_q = collections.deque()
    max_q = collections.deque()

    # Initialize the minimum difference to the maximum possible value (N-1).
    min_diff = N - 1

    # `v` is the 0-indexed value being processed (actual value is v+1).
    # We slide a window of values from {1..K} up to {N-K+1..N}.
    for v in range(N):
        current_pos = pos[v]

        # Update min_q: maintain elements in increasing order of their positions.
        while min_q and pos[min_q[-1]] >= current_pos:
            min_q.pop()
        min_q.append(v)

        # Update max_q: maintain elements in decreasing order of their positions.
        while max_q and pos[max_q[-1]] <= current_pos:
            max_q.pop()
        max_q.append(v)

        # Once the window has K elements, calculate the difference and update the minimum.
        if v >= K - 1:
            # The current window of 0-indexed values is [v - K + 1, v].
            # The value that just left the window is v - K.
            # If the heads of the deques correspond to this outdated value, remove them.
            if min_q[0] == v - K:
                min_q.popleft()
            if max_q[0] == v - K:
                max_q.popleft()

            # The front of min_q holds the value with the minimum position in the window.
            # The front of max_q holds the value with the maximum position in the window.
            min_pos_in_window = pos[min_q[0]]
            max_pos_in_window = pos[max_q[0]]
            
            current_diff = max_pos_in_window - min_pos_in_window
            min_diff = min(min_diff, current_diff)
            
    print(min_diff)

solve()