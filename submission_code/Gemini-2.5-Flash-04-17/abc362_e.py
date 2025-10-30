import sys

# Read input
N = int(sys.stdin.readline())
A = list(map(int, sys.stdin.readline().split()))

MOD = 998244353

# dp[i] is a map where keys are differences d, and values are lists of counts indexed by length k.
# dp[i][d][k] = number of arithmetic subsequences of length k ending at index i with common difference d.
# We use a list of maps for dp: dp[i] is the map for index i.
# dp[i] = { diff1: [0, count_len1, count_len2, ..., count_lenN], diff2: [...], ... }
# list index k corresponds to length k. We use lengths 1 to N.
# In this DP structure, length 1 subsequences (single elements) don't have a defined difference in dp[i].
# They serve as the base case for length 2 subsequences.
# The list size is N+1, using indices 1 to N for lengths.
dp = [{} for _ in range(N)]

# total_counts[k] = total number of arithmetic subsequences of length k.
# Initialize with counts for length 1.
total_counts = [0] * (N + 1)
total_counts[1] = N

# Iterate through the array to build up longer subsequences
# i is the index of the current element (the last element of the subsequence)
for i in range(N):
    # For each element A[i], consider all previous elements A[j] (j < i)
    # A[j] will be the second-to-last element of the subsequence ending at i.
    for j in range(i):
        # The common difference is determined by the last two elements.
        d = A[i] - A[j]

        # If this difference d is not seen before ending at index i,
        # initialize its counts list for index i.
        # dp[i][d] will store counts for lengths 2 to N ending at i with difference d.
        if d not in dp[i]:
            # Initialize list of size N+1. Index k will store count for length k.
            # List index 0 is unused, 1 is implicitly handled by base case.
            dp[i][d] = [0] * (N + 1)

        # 1. Consider forming a new subsequence of length 2: (A[j], A[i])
        # This sequence ends at index i with difference d.
        # It is formed by extending the length 1 subsequence (A[j]) ending at index j.
        # There is exactly 1 such length 1 subsequence (A[j]) ending at j.
        # This contributes 1 to the count of length 2 sequences ending at i with difference d.
        count_for_len_2 = 1
        dp[i][d][2] = (dp[i][d][2] + count_for_len_2) % MOD
        # Accumulate this count to the total count for length 2.
        total_counts[2] = (total_counts[2] + count_for_len_2) % MOD

        # 2. Consider extending arithmetic subsequences of length k-1 (where k-1 >= 2, so k >= 3)
        # that end at index j with the same difference d.
        # If such sequences existed ending at index j, use their counts from dp[j].
        if d in dp[j]: # Check if sequences ending at j with this difference existed
            prev_counts_at_j = dp[j][d]
            # Iterate through possible previous lengths k-1, starting from 2.
            # This forms sequences of length k, starting from 3.
            for k in range(3, N + 1):
                 # The count of length k-1 sequences ending at j with difference d is dp[j][d][k-1].
                count_at_j_len_k_minus_1 = prev_counts_at_j[k - 1]

                # If there were such previous sequences
                if count_at_j_len_k_minus_1 > 0:
                    # Extend these sequences by A[i]. They now become length k sequences ending at i with difference d.
                    dp[i][d][k] = (dp[i][d][k] + count_at_j_len_k_minus_1) % MOD
                    # Accumulate this count to the total count for length k.
                    total_counts[k] = (total_counts[k] + count_at_j_len_k_minus_1) % MOD

# The total_counts list now contains the final counts for each length k (from 1 to N).
# Print the results for k = 1, 2, ..., N separated by spaces.
results = [total_counts[k] for k in range(1, N + 1)]
print(*results)