import sys
import bisect

# Function to solve the problem
def solve():
    # Read N (number of sleighs) and Q (number of queries)
    N, Q = map(int, sys.stdin.readline().split())

    # Read the list of reindeer requirements for each sleigh
    R = list(map(int, sys.stdin.readline().split()))

    # To maximize the number of sleighs for a given number of reindeer,
    # we should choose the sleighs with the smallest requirements first.
    # So, sort the requirements in ascending order.
    R.sort()

    # Compute prefix sums of the sorted requirements.
    # PrefixSum[i] will store the total reindeer required to pull the first i
    # cheapest sleighs.
    # PrefixSum[0] is 0, representing the cost for 0 sleighs.
    # PrefixSum[1] is R[0], the cost for the cheapest sleigh.
    # PrefixSum[k] is sum(R[0]...R[k-1]), the cost for the first k cheapest sleighs.
    # The size of PrefixSum is N+1.
    PrefixSum = [0] * (N + 1)
    for i in range(N):
        PrefixSum[i + 1] = PrefixSum[i] + R[i]

    # Process each query
    for _ in range(Q):
        # Read the number of available reindeer for the current query
        X = int(sys.stdin.readline())

        # We want to find the maximum integer k such that PrefixSum[k] <= X.
        # This means we want the maximum number of sleighs (k) whose total requirement
        # is less than or equal to X.
        # The PrefixSum array is monotonically non-decreasing (strictly increasing since R_i >= 1).
        # We can use binary search.

        # bisect.bisect_right(a, x) finds an insertion point i such that
        # all elements in a[:i] are <= x and all elements in a[i:] are > x.
        # In our case, a is PrefixSum and x is X.
        # If bisect_right returns index `i`, it implies PrefixSum[k] <= X for k in [0, i-1] and
        # PrefixSum[k] > X for k in [i, N].
        # The indices k for which PrefixSum[k] <= X are 0, 1, ..., i-1.
        # The maximum such index is i-1, which corresponds to the maximum number of sleighs possible.
        
        # Find the index `i` where `X` would be inserted in `PrefixSum`
        # to maintain order, such that elements before `i` are <= `X`
        # and elements from `i` onwards are > `X`.
        insert_point = bisect.bisect_right(PrefixSum, X)
        
        # The number of sleighs corresponding to PrefixSum[k] is k.
        # If insert_point is i, it means PrefixSum[i-1] <= X is the last element
        # in the PrefixSum array that is less than or equal to X.
        # So, the maximum number of sleighs we can pull is i-1.
        max_sleighs = insert_point - 1

        # Print the result for the current query
        print(max_sleighs)

# Execute the solve function
solve()