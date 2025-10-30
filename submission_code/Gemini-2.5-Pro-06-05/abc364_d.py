# YOUR CODE HERE
import sys
import bisect

def solve():
    """
    Reads input, solves the problem, and prints the output.
    The solution uses binary search on the answer for each query.
    """
    # Read problem size N and number of queries Q.
    # Using sys.stdin.readline for faster I/O.
    try:
        line = sys.stdin.readline()
        if not line:
            return
        N, Q = map(int, line.split())
    except (ValueError, IndexError):
        # Handle cases with empty input lines.
        return

    # Read the N coordinates of points A and sort them. This preprocessing
    # step is crucial for efficient range counting later.
    A = list(map(int, sys.stdin.readline().split()))
    A.sort()

    # Process each of the Q queries.
    for _ in range(Q):
        # For each query, read the coordinate b and the rank k.
        b, k = map(int, sys.stdin.readline().split())

        # We need to find the k-th smallest distance from point b to a point in A.
        # This is solved by binary searching on the value of the distance itself.
        
        # 'low' and 'high' define the search space for the distance.
        low = 0
        high = 2 * 10**8  # A safe upper bound for the maximum possible distance.
        ans = high

        while low <= high:
            # `d` is the candidate distance we are checking.
            d = (low + high) // 2

            # Count how many points `a_i` are within distance `d` of `b`.
            # This is equivalent to counting `a_i` in the range `[b - d, b + d]`.
            # On a sorted array, this can be done in O(log N) using the bisect module.
            count = bisect.bisect_right(A, b + d) - bisect.bisect_left(A, b - d)

            if count >= k:
                # If `d` is large enough to include at least `k` points, it's a
                # potential answer. We store it and try to find a smaller valid distance.
                ans = d
                high = d - 1
            else:
                # If `d` is too small, we must search for a larger distance.
                low = d + 1
        
        # After the binary search, `ans` holds the k-th smallest distance.
        print(ans)

solve()