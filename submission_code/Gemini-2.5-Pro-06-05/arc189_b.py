import sys

def solve():
    """
    Reads problem input, computes the minimum possible sum of coordinates,
    and prints the result to standard output.
    """
    # Use a faster method for reading input, standard in competitive programming
    input = sys.stdin.readline
    
    try:
        # Read problem parameters
        N = int(input())
        X = list(map(int, input().split()))
    except (IOError, ValueError):
        # Gracefully handle empty or malformed input
        return

    # The operation allows swapping gaps d_i and d_{i+2}. This partitions the
    # gaps into two sets that can be permuted internally: {d_1, d_3, ...} and
    # {d_2, d_4, ...}.
    # The sum of coordinates is S = N*X_1 + sum_{j=1}^{N-1} (N-j)*d_j.
    # To minimize S, we pair the smallest gaps with the largest coefficients (N-j),
    # respecting the partition of gaps.

    # 1. Calculate the initial gaps between adjacent pieces.
    gaps = [X[i + 1] - X[i] for i in range(N - 1)]

    # 2. Separate gaps based on original index parity and sort them.
    # gaps[0::2] corresponds to d_1, d_3, ...
    # gaps[1::2] corresponds to d_2, d_4, ...
    d_odd_sorted = sorted(gaps[0::2])
    d_even_sorted = sorted(gaps[1::2])

    # 3. Calculate the minimized contribution from gaps to the total sum.
    total_gaps_contrib = 0

    # Contribution from odd-indexed gaps (j=1, 3, ...). The smallest gaps
    # are paired with the largest coefficients (N-1), (N-3), ...
    total_gaps_contrib += sum(
        (N - (2 * i + 1)) * g for i, g in enumerate(d_odd_sorted)
    )

    # Contribution from even-indexed gaps (j=2, 4, ...). The smallest gaps
    # are paired with the largest coefficients (N-2), (N-4), ...
    total_gaps_contrib += sum(
        (N - (2 * i + 2)) * g for i, g in enumerate(d_even_sorted)
    )

    # 4. Calculate the final minimum sum. The position of the first piece (X_1)
    # is invariant.
    min_sum = N * X[0] + total_gaps_contrib
    
    print(min_sum)

solve()