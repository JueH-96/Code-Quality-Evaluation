class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue
            for idx, word in enumerate(words):
                cost = costs[idx]
                m = len(word)
                if i + m > n:
                    continue
                if target[i:i+m] == word:
                    if dp[i] + cost < dp[i + m]:
                        dp[i + m] = dp[i] + cost
        
        return dp[n] if dp[n] != float('inf') else -1