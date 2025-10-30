import sys

def solve():
    # Read input N
    N = int(sys.stdin.readline())
    # Read input S
    S = sys.stdin.readline().strip()

    # Find 0-based indices of all '1's
    # Using list comprehension for efficiency
    one_indices = [i for i, char in enumerate(S) if char == '1']

    # Number of '1's
    k = len(one_indices)

    # If k is 0, which is guaranteed not to happen by constraints (S contains at least one 1),
    # but good practice to handle edge cases if constraints were different.
    # The problem guarantees k >= 1.
    # if k == 0:
    #     print(0)
    #     return

    # Calculate q_i = p'_i - i for i from 0 to k-1, where p'_i is the i-th 0-based index of '1'
    # q_values[i] = one_indices[i] - i
    # These q_values represent the deviation of each '1' from its ideal position
    # in a perfectly packed block starting at index 0.
    # The sequence q_values is sorted non-decreasingly because one_indices[i+1] >= one_indices[i] + 1
    # => one_indices[i+1] - (i+1) >= (one_indices[i] + 1) - (i+1) = one_indices[i] - i
    # => q_values[i+1] >= q_values[i]
    q_values = [one_indices[i] - i for i in range(k)]

    # The target positions for the '1's will be t', t'+1, ..., t'+k-1 (0-based)
    # The cost to move the i-th '1' from one_indices[i] to t' + i is |one_indices[i] - (t' + i)|
    # This can be rewritten as |(one_indices[i] - i) - t'| = |q_values[i] - t'|
    # We want to find the integer t' that minimizes the sum of absolute differences:
    # sum(|q_values[i] - t'|) over all i from 0 to k-1.
    # This sum is minimized when t' is the median of the q_values.

    # The median of a sorted list q_values of length k is the element at index k // 2 (integer division).
    # For odd k, this is the unique median. For even k, any value between q_values[k//2 - 1] and q_values[k//2] (inclusive)
    # minimizes the sum. Picking q_values[k//2] works.
    median_q = q_values[k // 2]

    # The minimum number of swaps required is the sum of the absolute differences
    # between each q_value and the median q_value. This is equivalent to sum(|p'_i - (t' + i)|)
    # where t' is the chosen median_q.
    total_swaps = sum(abs(q - median_q) for q in q_values)

    # Print the result
    print(total_swaps)

# Execute the solve function
solve()