import sys

# Function to solve a single test case
def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    if N == 0:
        sys.stdout.write("0
")
        return

    # Step 1: Compress adjacent identical elements into blocks
    # B stores the value of each block
    B = []
    if N > 0:
        B.append(A[0])
        for i in range(1, N):
            if A[i] != A[i-1]:
                B.append(A[i])

    M = len(B)

    # Step 2: Dynamic Programming
    # dp[i][j] = minimum cost to clear the sequence of blocks B[i...j]
    # using 0-based indexing for B
    # dp table is M x M
    # Initialize with a large value
    dp = [[float('inf')] * M for _ in range(M)]

    # Base case: intervals of length 1
    for i in range(M):
        dp[i][i] = 1 # Cost to delete a single block is 1 delete operation

    # Fill the DP table for intervals of length len from 2 to M
    for length in range(2, M + 1):
        for i in range(M - length + 1):
            j = i + length - 1

            # Option 1: Delete the first block B[i] alone
            # Cost = 1 (delete) + cost to clear remaining blocks B[i+1...j]
            # dp[i+1][j] is cost to clear B[i+1...j]. If i+1 > j, this is 0.
            dp[i][j] = 1 + dp[i+1][j]

            # Option 2: Find a later block B[k] (i < k <= j) with the same value as B[i]
            # We can clear the blocks between B[i] and B[k] (i.e., B[i+1...k-1]),
            # then solve the subproblem on B[k...j].
            # Cost = cost to clear B[i+1...k-1] + cost to clear B[k...j]
            # This recurrence implies that clearing B[i+1...k-1] makes B[i] and B[k]
            # effectively adjacent, and the cost of clearing B[i] (merged) + B[k...j]
            # is equivalent to the cost of clearing B[k...j] starting from B[k] (dp[k][j]).
            # The +1 for deleting B[i] is "saved" because it's merged with B[k]
            # and replaces the first deletion operation within the dp[k][j] subproblem.
            for k in range(i + 1, j + 1):
                if B[k] == B[i]:
                    # cost_intermediate is the cost to clear blocks B[i+1...k-1].
                    # This interval is empty if k-1 < i+1, i.e., k == i+1.
                    # dp[i+1][k-1] is 0 if (i+1) > (k-1).
                    # Since k >= i+1, the only case where i+1 > k-1 is when k = i+1.
                    # In this case, the interval is [i+1, i], which is empty, cost 0.
                    cost_intermediate = dp[i+1][k-1] if k > i + 1 else 0 # Correctly handles k = i+1

                    dp[i][j] = min(dp[i][j], cost_intermediate + dp[k][j])

    # The minimum cost to make A empty is the minimum cost to clear all blocks B[0...M-1]
    sys.stdout.write(str(dp[0][M-1]) + "
")


# Read the number of test cases
T = int(sys.stdin.readline())

# Process each test case
for _ in range(T):
    solve()