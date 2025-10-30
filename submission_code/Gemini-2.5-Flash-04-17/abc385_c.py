# YOUR CODE HERE
import sys

# Read N
n = int(sys.stdin.readline())

# Handle N=1 separately as the loops for DP might not run or dp table size calculation needs care
# If N=1, only one building exists, which forms a valid sequence of length 1.
if n == 1:
    print(1)
else:
    # Read heights H (adjusting to 0-based indexing)
    # Using sys.stdin.readline and strip().split() is generally faster for competitive programming
    h = list(map(int, sys.stdin.readline().strip().split()))

    # dp[j][d_idx] will store the length of the longest arithmetic progression sequence of buildings
    # ending at index j, with common difference d = d_idx + 1.
    # 0 <= j < n
    # 0 <= d_idx < n-1 (representing difference d from 1 to n-1)
    # Table size is n x (n-1).
    # Initialize with 0.
    # The conditional check `if n == 1:` handles the case where n-1 is 0, preventing creation of a 1x0 table.
    dp = [[0] * (n - 1) for _ in range(n)]

    # The maximum length is at least 1 (choosing a single building).
    max_length = 1

    # Iterate through the current ending index j
    # j goes from 0 to n-1
    for j in range(n):
        # Iterate through all possible common differences d (1 to n-1)
        # d_idx represents difference d = d_idx + 1.
        # The common difference d between index j and a previous index i (i < j) is j - i.
        # d can range from 1 (when i = j-1) up to j (when i = 0).
        # So d_idx can range from 0 (for d=1) up to j-1 (for d=j).
        # However, the maximum possible difference between any two indices in [0, N-1] is N-1.
        # So we need to consider all differences from 1 to N-1, which corresponds to d_idx from 0 to N-2.
        for d_idx in range(n - 1):
            d = d_idx + 1 # Common difference

            # Calculate the previous index in the arithmetic progression
            i = j - d

            # Check if the previous index i is valid (within the bounds [0, n-1])
            # and if the heights of buildings at index j and index i are the same.
            if i >= 0 and h[j] == h[i]:
                # The sequence ending at j with difference d extends a sequence
                # ending at i with the same difference d.
                # The length of the sequence ending at i with difference d is stored in dp[i][d_idx].

                # If a sequence of length >= 2 ending at index i with difference d already exists (dp[i][d_idx] > 0),
                # we extend it by adding building j. The new length is dp[i][d_idx] + 1.
                if dp[i][d_idx] > 0:
                    dp[j][d_idx] = dp[i][d_idx] + 1
                else:
                    # If there is no sequence of length >= 2 ending at i with difference d,
                    # this means the sequence consisting of buildings at indices i and j is the start
                    # of a new arithmetic progression of length 2 with difference d.
                    dp[j][d_idx] = 2

                # Update the maximum length found so far across all ending indices and differences.
                max_length = max(max_length, dp[j][d_idx])

    # Print the final maximum length found.
    print(max_length)