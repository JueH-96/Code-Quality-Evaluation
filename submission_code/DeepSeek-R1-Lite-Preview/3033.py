class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        # Find positions where s1 and s2 differ
        mismatches = [i for i in range(n) if s1[i] != s2[i]]
        k = len(mismatches)
        
        # If the number of mismatches is odd, it's impossible
        if k % 2 != 0:
            return -1
        
        # If there are no mismatches, cost is 0
        if k == 0:
            return 0
        
        # DP array to store minimum cost to cover first i mismatches
        dp = [float('inf')] * (k + 1)
        dp[0] = 0  # Base case: no mismatches, cost is 0
        
        # Fill the DP array
        for i in range(1, k + 1):
            for j in range(i):
                if j < i - 1:
                    # Pair mismatches[j] and mismatches[i-1]
                    cost = min(x, mismatches[i-1] - mismatches[j])
                    dp[i] = min(dp[i], dp[j] + cost)
        
        return dp[k] if dp[k] != float('inf') else -1