import sys

def solve():
    # Read input
    N = int(sys.stdin.readline())
    X = list(map(int, sys.stdin.readline().split()))

    # The initial coordinates are already sorted: X_1 < X_2 < ... < X_N
    # Let p_1, ..., p_N be the coordinates in ascending order at any state.
    # Let d_j = p_{j+1} - p_j for j = 1, ..., N-1 be the differences. d_j > 0.
    # The total sum of coordinates is S = sum_{k=1}^N p_k.
    # We can express p_k in terms of p_1 and differences: p_k = p_1 + sum_{j=1}^{k-1} d_j for k > 1.
    # The sum S can be rewritten as:
    # S = p_1 + sum_{k=2}^N (p_1 + sum_{j=1}^{k-1} d_j)
    # S = N * p_1 + sum_{k=2}^N sum_{j=1}^{k-1} d_j
    # Rearranging the inner sum by difference index j:
    # For a fixed d_j, it appears in the sum for p_k where k-1 >= j, i.e., k > j.
    # These terms are p_{j+1}, p_{j+2}, ..., p_N. There are N - (j+1) + 1 = N - j such terms.
    # So the coefficient of d_j in the total sum is (N - j).
    # S = N * p_1 + sum_{j=1}^{N-1} (N-j) * d_j.

    # The operation involves the i-th, (i+1)-th, (i+2)-th, (i+3)-th pieces in ascending order (1 <= i <= N-3).
    # Let their coordinates be p_i, p_{i+1}, p_{i+2}, p_{i+3}.
    # The operation replaces p_{i+1} with p_i + p_{i+3} - p_{i+1} and p_{i+2} with p_i + p_{i+3} - p_{i+2}.
    # It can be shown that the new sorted coordinates in this window are p_i, p_i+p_{i+3}-p_{i+2}, p_i+p_{i+3}-p_{i+1}, p_{i+3}.
    # Let d'_j be the new differences after the operation.
    # d'_i = (p_i + p_{i+3} - p_{i+2}) - p_i = p_{i+3} - p_{i+2} = d_{i+2}
    # d'_{i+1} = (p_i + p_{i+3} - p_{i+1}) - (p_i + p_{i+3} - p_{i+2}) = p_{i+2} - p_{i+1} = d_{i+1}
    # d'_{i+2} = p_{i+3} - (p_i + p_{i+3} - p_{i+1}) = p_{i+1} - p_i = d_i
    # Differences d_j for j < i or j > i+2 are unchanged by this specific operation application.
    # The operation at index i effectively swaps the values of d_i and d_{i+2} in the sequence of differences, keeping d_{i+1} and others outside {i, i+1, i+2} fixed.

    # The allowed operations are for 1 <= i <= N-3.
    # i=1 swaps d_1 and d_3.
    # i=2 swaps d_2 and d_4.
    # i=3 swaps d_3 and d_5.
    # ...
    # i=N-3 swaps d_{N-3} and d_{N-1}.
    # This means any two differences d_j and d_k can be swapped if j and k have the same parity and |j-k| is an even number >= 2.
    # This implies differences with odd indices {d_1, d_3, d_5, ...} can be permuted arbitrarily among themselves using a series of adjacent swaps (d_i, d_{i+2}).
    # Differences with even indices {d_2, d_4, d_6, ...} can be permuted arbitrarily among themselves using a series of adjacent swaps (d_i, d_{i+2}).
    # The minimum coordinate p_1 is always the initial X_1, as the operation replaces elements only strictly between p_i and p_{i+3}. When i=1, p_1 is not replaced.

    # To minimize S = N * X_1 + sum_{j=1}^{N-1} (N-j) * d_j, where X_1 is fixed.
    # We need to assign the initial differences {X_2-X_1, ..., X_N-X_{N-1}} to the current differences {d_1, ..., d_{N-1}}.
    # The coefficients (N-j) are strictly decreasing for j = 1, ..., N-1.
    # To minimize sum_{j=1}^{N-1} (N-j) * d_j, we should pair the smallest available differences with the largest coefficients (smallest j) within their respective parity groups.
    # The odd-indexed differences {d_1, d_3, d_5, ...} should take values from the sorted list of initial odd-indexed differences.
    # The even-indexed differences {d_2, d_4, d_6, ...} should take values from the sorted list of initial even-indexed differences.

    # Calculate initial differences: d_j = X_{j+1} - X_j for j=1, ..., N-1.
    initial_diffs = [X[i+1] - X[i] for i in range(N-1)] # initial_diffs[k] corresponds to d_{k+1}

    # Separate odd and even indexed initial differences
    odd_indexed_initial_diffs = []
    even_indexed_initial_diffs = []
    for k in range(N-1): # k is 0-based index of initial_diffs list
        j = k + 1 # j is 1-based index of the difference d_j
        if j % 2 != 0: # d_1, d_3, d_5, ...
            odd_indexed_initial_diffs.append(initial_diffs[k])
        else: # d_2, d_4, d_6, ...
            even_indexed_initial_diffs.append(initial_diffs[k])

    # Sort the difference lists in ascending order
    odd_indexed_initial_diffs.sort()
    even_indexed_initial_diffs.sort()

    # Construct the optimal differences d_j^* and calculate the minimum sum
    # Minimum sum S = N * X_1 + sum_{j=1}^{N-1} (N-j) * d_j^*
    # X_1 is the first element of the sorted initial array X.
    min_sum = N * X[0]

    odd_sorted_idx = 0
    even_sorted_idx = 0

    # Iterate through the difference indices j from 1 to N-1
    for j in range(1, N): # j is the 1-based index of the difference d_j^*
        coeff = N - j
        if j % 2 != 0: # d_j^* for odd j (1, 3, 5, ...) gets smallest available odd difference
            d_j_star = odd_indexed_initial_diffs[odd_sorted_idx]
            odd_sorted_idx += 1
        else: # d_j^* for even j (2, 4, 6, ...) gets smallest available even difference
            d_j_star = even_indexed_initial_diffs[even_sorted_idx]
            even_sorted_idx += 1
        min_sum += coeff * d_j_star

    print(min_sum)

solve()