class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        # Collect the indices where the two strings differ
        diffs = [i for i in range(len(s1)) if s1[i] != s2[i]]
        n = len(diffs)
        
        # If the number of differing indices is odd, it's impossible
        if n % 2 != 0:
            return -1
        
        if n == 0:
            return 0
        
        # Initialize the DP table with infinity
        INF = float('inf')
        dp = [[INF] * n for _ in range(n)]
        
        # Fill the DP table
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if (j - i + 1) % 2 != 0:
                    continue  # Skip if the segment length is odd
                if i == j:
                    dp[i][j] = 0  # No cost for a single element (though this case won't be used)
                elif j == i + 1:
                    # Calculate the cost for pairing these two adjacent elements
                    cost = min(x, diffs[j] - diffs[i])
                    dp[i][j] = cost
                else:
                    # Consider all possible ways to split the segment into two parts
                    min_cost = INF
                    for k in range(i, j):
                        # Ensure both parts have even lengths
                        if (k - i + 1) % 2 == 0 and (j - (k + 1) + 1) % 2 == 0:
                            current_cost = dp[i][k] + dp[k + 1][j]
                            if current_cost < min_cost:
                                min_cost = current_cost
                    if min_cost != INF:
                        dp[i][j] = min_cost
        
        # The minimal cost is in dp[0][n-1]
        if dp[0][n - 1] == INF:
            return -1
        else:
            return dp[0][n - 1]