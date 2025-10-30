class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(n):
            if dp[i] == float('inf'):
                continue
            for j in range(len(words)):
                word = words[j]
                cost = costs[j]
                if target[i:min(i + len(word), n + 1)] == word:
                    dp[i + len(word)] = min(dp[i + len(word)], dp[i] + cost)
        if dp[n] == float('inf'):
            return -1
        else:
            return dp[n]