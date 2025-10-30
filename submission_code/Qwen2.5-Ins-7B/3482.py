from typing import List

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        dp = [float('inf')] * (len(target) + 1)
        dp[0] = 0
        
        for i in range(1, len(target) + 1):
            for j in range(len(words)):
                if target[i-1] in words[j]:
                    start = target.rfind(words[j], 0, i-1)
                    if start == -1:
                        dp[i] = min(dp[i], dp[start] + costs[j])
        
        return dp[-1] if dp[-1] != float('inf') else -1