# YOUR CODE HERE
import sys

def solve():
    """
    Reads input, solves the problem, and prints the result to standard output.
    """
    # Read N from the first line of stdin. It's part of the input format
    # but not strictly necessary for the algorithm as len(S) can be used.
    try:
        N = int(sys.stdin.readline())
        S = sys.stdin.readline().strip()
    except (IOError, ValueError):
        # Handle cases with malformed or empty input.
        return

    # Find all 0-based indices of '1's in the string.
    one_indices = [i for i, char in enumerate(S) if char == '1']

    # Get the count of '1's.
    k = len(one_indices)

    # The problem guarantees at least one '1' (k >= 1).
    # If there's only one '1', it's already a contiguous block, so no swaps are needed.
    if k <= 1:
        print(0)
        return

    # As derived in the explanation, we compute the sequence c_i = p_i - i.
    c = [p_val - i for i, p_val in enumerate(one_indices)]

    # The optimal target block starting position `l` is the median of `c`.
    # Since `c` is sorted (non-decreasing), its median is the element at index k // 2.
    median_l = c[k // 2]

    # The minimum cost is the sum of absolute differences from the median.
    cost = sum(abs(val - median_l) for val in c)

    # Print the final answer.
    print(cost)

# Execute the solution
solve()