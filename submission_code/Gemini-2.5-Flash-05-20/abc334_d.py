import sys
import bisect

def solve():
    # Read N and Q
    N, Q = map(int, sys.stdin.readline().split())

    # Read R values and convert to a list of integers
    R = list(map(int, sys.stdin.readline().split()))

    # Sort R values in ascending order to prioritize sleighs requiring fewer reindeer
    R.sort()

    # Calculate prefix sums
    # prefix_sums[k] will store the minimum total reindeer required to pull k sleighs
    # prefix_sums[0] = 0 (for 0 sleighs)
    # prefix_sums[k] = sum of R_sorted[0]...R_sorted[k-1]
    prefix_sums = [0] * (N + 1)
    current_sum = 0
    for i in range(N):
        current_sum += R[i]
        prefix_sums[i+1] = current_sum

    # Process each query
    results = []
    for _ in range(Q):
        X = int(sys.stdin.readline())

        # Use bisect_right to find the largest k such that prefix_sums[k] <= X
        # bisect_right returns an insertion point 'idx'.
        # This 'idx' means that all elements prefix_sums[0] to prefix_sums[idx-1]
        # are less than or equal to X.
        # Thus, (idx - 1) is the maximum number of sleighs that can be pulled.
        k = bisect.bisect_right(prefix_sums, X) - 1
        results.append(str(k))

    # Print all results, each on a new line
    sys.stdout.write("
".join(results) + "
")

# Call the solve function to run the program
solve()