import sys
import bisect

def solve():
    # Read N
    N = int(sys.stdin.readline())

    # Read X coordinates. X_1 ... X_N are guaranteed to be sorted.
    X = list(map(int, sys.stdin.readline().split()))

    # Read P values (villagers)
    P = list(map(int, sys.stdin.readline().split()))

    # Precompute prefix sums of villagers
    # prefix_sums[i] will store the sum of P[0]...P[i-1].
    # prefix_sums[0] = 0 (sum of an empty prefix).
    # The length of prefix_sums will be N + 1.
    prefix_sums = [0] * (N + 1)
    for i in range(N):
        prefix_sums[i+1] = prefix_sums[i] + P[i]

    # Read Q (number of queries)
    Q = int(sys.stdin.readline())

    # Use a list to store results and print them all at once for efficiency.
    results = []

    # Process each query
    for _ in range(Q):
        L, R = map(int, sys.stdin.readline().split())

        # Find the index of the first village whose coordinate is >= L.
        # bisect_left returns an insertion point which comes before (to the left of)
        # any existing entries of x in a.
        # This means X[start_idx] is the first element in X that is >= L.
        start_idx = bisect.bisect_left(X, L)

        # Find the index of the first village whose coordinate is > R.
        # bisect_right returns an insertion point which comes after (to the right of)
        # any existing entries of x in a.
        # This means all elements X[j] where j < end_idx are <= R.
        end_idx = bisect.bisect_right(X, R)

        # The villages we want to sum are those with original indices from start_idx to end_idx - 1.
        # The sum of villagers for this range is:
        # P[start_idx] + P[start_idx + 1] + ... + P[end_idx - 1]
        # Using prefix sums, this is equivalent to: prefix_sums[end_idx] - prefix_sums[start_idx].
        current_sum = prefix_sums[end_idx] - prefix_sums[start_idx]
        
        results.append(str(current_sum))
    
    # Print all results, each on a new line, followed by a final newline.
    sys.stdout.write("
".join(results) + "
")

# This ensures that solve() is called only when the script is executed directly.
if __name__ == '__main__':
    solve()