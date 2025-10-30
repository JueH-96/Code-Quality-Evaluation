class Solution:
    def minValidStrings(self, words, target):
        prefix = set()
        for word in words:
            for i in range(len(word)):
                prefix.add(word[:i+1])
        dp = [0] + [float('inf')] * len(target)
        for i in range(1, len(target) + 1):
            for j in range(i):
                if target[j:i] in prefix:
                    dp[i] = min(dp[i], dp[j] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1