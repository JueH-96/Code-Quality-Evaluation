class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        differing_indices = []
        for i, (c1, c2) in enumerate(zip(s1, s2)):
            if c1 != c2:
                differing_indices.append(i)
        m = len(differing_indices)
        if m % 2 != 0:
            return -1
        if m == 0:
            return 0
        
        n = m
        dp = [[0] * n for _ in range(n)]
        
        for length in range(2, n + 1, 2):
            for i in range(n - length + 1):
                j = i + length - 1
                if length == 2:
                    cost = min(differing_indices[j] - differing_indices[i], x)
                    dp[i][j] = cost
                else:
                    min_cost = float('inf')
                    # Iterate k in steps of 2 to ensure (k - i) is odd
                    for k in range(i + 1, j + 1, 2):
                        current_cost = min(differing_indices[k] - differing_indices[i], x) 
                        current_cost += dp[i + 1][k - 1]
                        if k + 1 <= j:
                            current_cost += dp[k + 1][j]
                        if current_cost < min_cost:
                            min_cost = current_cost
                    dp[i][j] = min_cost
        
        return dp[0][n - 1]