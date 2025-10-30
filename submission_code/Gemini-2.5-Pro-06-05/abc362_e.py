# YOUR CODE HERE
import sys
import collections

def solve():
    """
    This function reads the input, solves the arithmetic subsequences problem
    using dynamic programming, and prints the result.
    """
    try:
        N = int(sys.stdin.readline())
        if N == 0:
            return
        A = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # In a contest setting, assume valid input. This handles local testing issues.
        return

    MOD = 998244353

    # ans[k] will store the total number of arithmetic subsequences of length k.
    # It is updated incrementally as we discover new subsequences.
    ans = [0] * (N + 1)

    # dp[j][d][k] stores the number of arithmetic subsequences of length k
    # ending at index j with common difference d.
    # To implement this efficiently, dp[j] is a defaultdict mapping d to a list of counts.
    dp = [collections.defaultdict(lambda: [0] * (N + 1)) for _ in range(N)]

    # Base case: All subsequences of length 1 are arithmetic.
    if N > 0:
        ans[1] = N

    # Iterate through all pairs of indices (i, j) with i < j
    for j in range(N):
        for i in range(j):
            d = A[j] - A[i]

            # The pair (A[i], A[j]) forms a new AP of length 2.
            # This is equivalent to extending a length-1 subsequence (just A[i]).
            count_len2 = 1
            dp[j][d][2] = (dp[j][d][2] + count_len2) % MOD
            ans[2] = (ans[2] + count_len2) % MOD

            # Extend APs of length k >= 2 that ended at index i with the same difference.
            counts_at_i = dp[i][d]
            for k in range(2, N):
                count_to_extend = counts_at_i[k]
                if count_to_extend > 0:
                    new_len = k + 1
                    dp[j][d][new_len] = (dp[j][d][new_len] + count_to_extend) % MOD
                    ans[new_len] = (ans[new_len] + count_to_extend) % MOD

    # Print the results for k = 1 to N, separated by spaces.
    print(*ans[1:])


# Execute the solution
solve()