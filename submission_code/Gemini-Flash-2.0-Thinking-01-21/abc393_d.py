import sys

def solve():
    # Read N
    N = int(sys.stdin.readline())
    # Read S
    S = sys.stdin.readline().strip()

    # 1. Find indices of '1's (0-indexed)
    one_indices = [i for i, char in enumerate(S) if char == '1']
    k = len(one_indices) # k is the number of '1's

    # If there are no '1's or all characters are '1's, they are already contiguous.
    # Problem guarantees at least one '1', so k >= 1.
    # If k == N, all characters are '1's, already contiguous.
    # The code below correctly handles k=1 and k=N, resulting in a cost of 0.
    # For k=1, d_values has one element, median is that element, sum is 0.
    # For k=N, one_indices = [0, 1, ..., N-1]. d_values = [0, 0, ..., 0]. median is 0, sum is 0.

    # 2. Calculate d_p = idx_p - p for p = 0, ..., k-1
    # idx_p is the original 0-based index of the p-th '1' (0-indexed).
    # If the contiguous block starts at 0-based index j, the p-th '1' should go to target index j + p.
    # The number of swaps for the p-th '1' is |idx_p - (j + p)|.
    # Total swaps = sum_{p=0}^{k-1} |idx_p - (j + p)| = sum_{p=0}^{k-1} |(idx_p - p) - j|.
    # Let d_p = idx_p - p. We want to minimize sum_{p=0}^{k-1} |d_p - j| by choosing integer j.
    # This sum is minimized when j is the median of the set {d_0, d_1, ..., d_{k-1}}.
    # The list d_values = [idx_p - p for p in range(k)] is sorted because idx_p - p is non-decreasing.
    # (If p < q, idx_p < idx_q. idx_q - idx_p >= q - p. (idx_q - q) - (idx_p - p) = (idx_q - idx_p) - (q - p) >= 0).
    d_values = [one_indices[p] - p for p in range(k)]

    # 3. Find the median of d_values. Since d_values is sorted, the median value is d_values[k//2].
    # k//2 gives the floor of k/2, which is the correct 0-based index for the median element.
    median_d = d_values[k // 2]

    # 4. Calculate the minimum number of swaps: sum_{p=0}^{k-1} |d_values[p] - median_d|
    # This is the sum of absolute deviations from the median, which corresponds to the minimum swaps.
    cost = 0
    for d_val in d_values:
        cost += abs(d_val - median_d)

    # Print the result
    print(cost)

solve()