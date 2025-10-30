class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        m, n = len(target), len(words)
        dp = [0] + [float('inf')] * m
        for i in range(1, m + 1):
            for j in range(n):
                word = words[j]
                c = costs[j]
                k = i
                while k > 0 and target[k - 1] == word[-1]:
                    k -= 1
                if k == 0:
                    dp[i] = min(dp[i], dp[i - 1] + c)
                else:
                    dp[i] = min(dp[i], dp[k - 1] + c)
        return dp[-1] if dp[-1] < float('inf') else -1