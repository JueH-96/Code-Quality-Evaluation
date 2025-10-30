# YOUR CODE HERE
import sys

def solve():
    # Read input
    N, K = map(int, sys.stdin.readline().split())
    # A is 1-indexed in problem statement, 0-indexed here.
    # A_1, A_2, ..., A_K becomes A[0], A[1], ..., A[K-1]
    A = list(map(int, sys.stdin.readline().split())) 

    # The socks not in A form N-K pairs of the same color, contributing 0 weirdness.
    # We only need to consider the K socks with colors in A, each having 1 sock.

    # Total socks = 2N - K
    # Total pairs to form = floor((2N - K) / 2)
    # Pairs formed from 2-sock colors = N - K (these have 0 weirdness)
    # Pairs to form from K 1-sock colors = floor((2N - K) / 2) - (N - K)

    # Let M be the number of pairs to form from K 1-sock colors.
    # If 2N-K is even, K is even. M = (2N-K)/2 - (N-K) = K/2.
    # If 2N-K is odd, K is odd. M = (2N-K-1)/2 - (N-K) = (K-1)/2.
    # In both cases, M = K // 2.

    # We have K socks with colors A[0], ..., A[K-1].
    # We need to form M = K // 2 pairs from these K socks, minimizing weirdness.
    # This is equivalent to selecting K - 2M = K % 2 socks to leave unpaired.
    # If K is even, K%2 = 0. M = K/2. We must pair all K socks into K/2 pairs.
    # If K is odd, K%2 = 1. M = (K-1)/2. We must leave 1 sock unpaired, and form M pairs from the remaining K-1 socks.

    if K % 2 == 0:
        # K is even. We must pair all K socks (A[0]...A[K-1]) into K/2 pairs.
        # The minimum weirdness is achieved by pairing adjacent elements: (A[0], A[1]), (A[2], A[3]), ..., (A[K-2], A[K-1]).
        total_weirdness = 0
        for i in range(K // 2):
            total_weirdness += A[2 * i + 1] - A[2 * i]
        print(total_weirdness)
    else:
        # K is odd. K = 2M + 1, M = (K-1)//2.
        # We must leave one sock unpaired from A[0]...A[K-1].
        # We need to find the minimum weirdness achieved by leaving each possible sock A[idx] (0-indexed idx) unpaired.
        # When A[idx] is left unpaired, the remaining K-1 elements are paired optimally (adjacently in the sorted sublist).

        # Calculate adjacent differences of A
        # diffs[i] = A[i+1] - A[i] for i = 0...K-2. Length K-1.
        diffs = [A[i+1] - A[i] for i in range(K-1)]

        # Calculate prefix sums of differences at even indices (0, 2, 4, ...) in diffs list.
        # p_even_idx_diffs[i] = diffs[0] + diffs[2] + ... + diffs[2*i - 2] for i >= 1. p_even_idx_diffs[0] = 0.
        # p_even_idx_diffs has size M+1, indices 0..M. Corresponds to p_odd from derivation.
        p_even_idx_diffs = [0] * (M + 1)
        for i in range(1, M + 1):
            p_even_idx_diffs[i] = p_even_idx_diffs[i-1] + diffs[2 * i - 2]

        # Calculate prefix sums of differences at odd indices (1, 3, 5, ...) in diffs list.
        # p_odd_idx_diffs[i] = diffs[1] + diffs[3] + ... + diffs[2*i - 1] for i >= 1. p_odd_idx_diffs[0] = 0.
        # p_odd_idx_diffs has size M+1, indices 0..M. Corresponds to p_even from derivation.
        p_odd_idx_diffs = [0] * (M + 1)
        for i in range(1, M + 1):
            p_odd_idx_diffs[i] = p_odd_idx_diffs[i-1] + diffs[2 * i - 1]

        min_weirdness = float('inf') # Use float('inf') for initialization

        # Iterate through each possible sock A[idx] (0-indexed idx) to leave unpaired.
        # Case 1: Leave A[idx] where idx is even, idx = 2*i. Corresponds to 1-based index 2i+1 (odd).
        # i ranges from 0 to M.
        # The weirdness is sum_{k=0}^{i-1} diffs[2k] + sum_{k=i}^{M-1} diffs[2k+1].
        # Sum_{k=0}^{i-1} diffs[2k] = p_even_idx_diffs[i].
        # Sum_{k=i}^{M-1} diffs[2k+1] = p_odd_idx_diffs[M] - p_odd_idx_diffs[i].
        for i in range(M + 1):
            weirdness = p_even_idx_diffs[i] + (p_odd_idx_diffs[M] - p_odd_idx_diffs[i])
            min_weirdness = min(min_weirdness, weirdness)

        # Case 2: Leave A[idx] where idx is odd, idx = 2*i - 1. Corresponds to 1-based index 2i (even).
        # i ranges from 1 to M.
        # The weirdness is sum_{k=0}^{i-2} diffs[2k] + (diffs[2i-2] + diffs[2i-1]) + sum_{k=i}^{M-1} diffs[2k+1].
        # Sum_{k=0}^{i-2} diffs[2k] = p_even_idx_diffs[i-1].
        # Sum_{k=i}^{M-1} diffs[2k+1] = p_odd_idx_diffs[M] - p_odd_idx_diffs[i].
        for i in range(1, M + 1):
            # idx = 2*i - 1
            weirdness = p_even_idx_diffs[i-1] + (diffs[2*i - 2] + diffs[2*i - 1]) + (p_odd_idx_diffs[M] - p_odd_idx_diffs[i])
            min_weirdness = min(min_weirdness, weirdness)

        print(min_weirdness)

solve()