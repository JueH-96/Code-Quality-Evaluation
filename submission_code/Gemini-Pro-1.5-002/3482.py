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
                k = 0
                l = i
                while k < len(word) and l < n and word[k] == target[l]:
                    k += 1
                    l += 1
                if k == len(word):
                    dp[l] = min(dp[l], dp[i] + cost)
        if dp[n] == float('inf'):
            return -1
        return dp[n]