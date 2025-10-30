import sys
from collections import defaultdict

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353

    # dp[i][diff] will be a list where dp[i][diff][k] stores
    # the number of arithmetic subsequences of length k
    # ending at index i with common difference 'diff'.
    # Using defaultdict for dp[i] to handle new differences gracefully.
    dp = [defaultdict(lambda: [0] * (N + 1)) for _ in range(N)]

    # ans[k] will store the total count of arithmetic subsequences of length k.
    # ans[0] is unused, ans[1] to ans[N] store the results.
    ans = [0] * (N + 1)

    # Base case: All single elements A[i] form an arithmetic subsequence of length 1.
    for i in range(N):
        ans[1] = (ans[1] + 1) % MOD

    # Iterate through each element A[i] to consider it as the last element of an AP.
    for i in range(N):
        # Iterate through each element A[j] before A[i] to consider it as the second to last element.
        for j in range(i):
            diff = A[i] - A[j]

            # Case 1: Forming an arithmetic subsequence of length 2.
            # The pair (A[j], A[i]) forms one such subsequence.
            # Increment the count for length 2 for subsequences ending at A[i] with 'diff'.
            dp[i][diff][2] = (dp[i][diff][2] + 1) % MOD
            # Add this to the total count for length 2.
            ans[2] = (ans[2] + 1) % MOD

            # Case 2: Extending existing arithmetic subsequences.
            # Look at subsequences ending at A[j] with the same 'diff'.
            # If a subsequence of length 'k_prev' ends at A[j] with 'diff',
            # then appending A[i] creates a new subsequence of length 'k_prev + 1'.
            # We iterate k_prev from 2 up to N-1, as k_prev+1 can go up to N.
            for k_prev in range(2, N):
                count_k_prev = dp[j][diff][k_prev]
                if count_k_prev > 0:
                    # Update dp state for A[i] with 'diff' for the new length (k_prev + 1).
                    dp[i][diff][k_prev + 1] = (dp[i][diff][k_prev + 1] + count_k_prev) % MOD
                    # Add this count to the overall answer for length (k_prev + 1).
                    ans[k_prev + 1] = (ans[k_prev + 1] + count_k_prev) % MOD

    # Print the answers for k = 1, 2, ..., N.
    # ans[1:] slices the list from index 1 to N, excluding ans[0].
    print(*(ans[1:]))

# Call the solve function to run the program.
solve()