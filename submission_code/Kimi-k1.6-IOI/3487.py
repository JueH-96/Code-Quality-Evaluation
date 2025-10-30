class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        target_set = set(targetIndices)
        m = len(pattern)
        n = len(source)
        dp = [float('inf')] * (m + 1)
        dp[0] = 0
        
        for j in range(n):
            current_char = source[j]
            # Update the DP array in reverse to avoid using the same character multiple times in one step
            for i in range(m - 1, -1, -1):
                if current_char == pattern[i]:
                    cost = 1 if j in target_set else 0
                    if dp[i] + cost < dp[i + 1]:
                        dp[i + 1] = dp[i] + cost
        
        # The minimal number of target indices used is dp[m]
        # The answer is the total target indices minus this minimal number
        return len(targetIndices) - dp[m] if dp[m] != float('inf') else 0