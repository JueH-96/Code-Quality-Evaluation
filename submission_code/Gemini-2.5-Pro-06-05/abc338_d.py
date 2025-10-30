# YOUR CODE HERE
import sys

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        # Read N (number of islands) and M (number of tour stops)
        n, m = map(int, sys.stdin.readline().split())
        # Read the sequence of M islands to visit
        x = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # This case should not occur based on problem constraints.
        return

    # l_base will store the sum of shortest path lengths in the original cycle.
    # This is the tour length if no bridges were "inconveniently" chosen.
    l_base = 0
    
    # We use a difference array to efficiently calculate the total penalty 
    # for closing each bridge. The penalty is the sum of increases in path lengths
    # over all segments of the tour.
    # The array is 1-indexed for bridges 1 to N. Size N+2 avoids out-of-bounds.
    diff_array = [0] * (n + 2)

    # Process each of the M-1 segments of the tour.
    for i in range(m - 1):
        u, v = x[i], x[i+1]
        
        # To have a consistent way of defining paths, let u < v.
        if u > v:
            u, v = v, u

        # In a cycle of N nodes, there are two paths between u and v:
        # 1. The "direct" path: u -> u+1 -> ... -> v
        # 2. The "wrap-around" path: v -> ... -> N -> 1 -> ... -> u
        path1_len = v - u
        path2_len = n - path1_len

        # The base length for this segment is the shorter of the two paths.
        l_base += min(path1_len, path2_len)
        
        # The penalty is the extra distance one has to travel if the shortest
        # path is blocked. This is the difference between the long and short paths.
        penalty = abs(path1_len - path2_len)

        # If penalty is 0, both paths have the same length, so closing any bridge
        # does not increase the travel distance for this segment.
        if penalty == 0:
            continue

        if path1_len <= path2_len:
            # Shortest path is the direct one, which uses bridges u, u+1, ..., v-1.
            # Closing any of these bridges incurs the penalty. We perform a range update
            # on the penalty counts for the range of bridges [u, v-1].
            diff_array[u] += penalty
            diff_array[v] -= penalty
        else: # path2_len < path1_len
            # Shortest path is the wrap-around one, using bridges 1..u-1 and v..N.
            # We perform two range updates: for [1, u-1] and [v, N].
            diff_array[1] += penalty
            diff_array[u] -= penalty
            diff_array[v] += penalty
            # The corresponding `diff_array[n + 1] -= penalty` is not needed as our
            # prefix sum calculation loop only goes up to n.

    # Reconstruct the actual penalty for each bridge by taking prefix sums of the
    # difference array.
    min_penalty = float('inf')
    current_penalty = 0
    for i in range(1, n + 1):
        current_penalty += diff_array[i]
        min_penalty = min(min_penalty, current_penalty)

    # The minimum possible tour length is the base length plus the minimum
    # penalty we can achieve by choosing the optimal bridge to close.
    final_answer = l_base + min_penalty

    # Print the result.
    print(final_answer)

solve()