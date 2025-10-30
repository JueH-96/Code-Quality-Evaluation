# YOUR CODE HERE
import sys
import bisect
from itertools import accumulate

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        # Read problem parameters N (number of sleighs) and Q (number of queries)
        # using fast I/O.
        line = sys.stdin.readline()
        if not line:
            return
        n, q = map(int, line.split())
        
        # Read the reindeer requirements for each sleigh.
        r = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        # Handle cases with malformed or empty input lines gracefully.
        return

    # The greedy strategy: to maximize the number of sleighs, we should always pick
    # the ones that require the fewest reindeer.
    # We sort the reindeer requirements to easily pick the "cheapest" ones.
    r.sort()

    # To quickly answer queries about the cost of pulling `k` sleighs,
    # we precompute the prefix sums of the sorted requirements.
    # `prefix_sums[k]` will hold the sum of the first `k` elements of sorted `r`,
    # which is the minimum cost to pull `k` sleighs.
    # We prepend a 0 to represent the cost of pulling 0 sleighs (which is 0).
    # `itertools.accumulate` is a clean and efficient way to compute this.
    prefix_sums = [0] + list(accumulate(r))

    # Process each query.
    for _ in range(q):
        # Read the available number of reindeer for the current query.
        x = int(sys.stdin.readline())

        # We need to find the largest `k` such that `prefix_sums[k] <= x`.
        # Since `prefix_sums` is a monotonically increasing array, this is a
        # classic binary search problem.
        # The `bisect` module in Python provides an efficient implementation.
        
        # `bisect.bisect_right(a, x)` finds an insertion point `i` which comes
        # after (to the right of) any existing entries of `x` in `a`.
        # This means that for all indices `j < i`, `a[j] <= x`.
        # The indices of our `prefix_sums` array (0, 1, ..., N) correspond to
        # the number of sleighs. Thus, the maximum number of sleighs we can
        # afford is `i - 1`.
        ans = bisect.bisect_right(prefix_sums, x) - 1
        
        # Output the result for the query.
        print(ans)

# Run the solution
solve()