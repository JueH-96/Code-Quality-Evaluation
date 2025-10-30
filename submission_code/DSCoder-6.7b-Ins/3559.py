class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        m, n = len(words), len(target)
        dp = [0] + [-1]*n
        for i in range(m):
            temp = [0] + dp[:]
            for j in range(n):
                if dp[j] != -1:
                    for k in range(1, min(j+1, len(words[i]))+1):
                        if temp[j-k+1] == -1 or temp[j-k+1] > dp[j] + words[i][:k] != target[j-k+1:j+1]:
                            temp[j-k+1] = dp[j] + (words[i][:k] == target[j-k+1:j+1])
            dp = temp
        return max(dp)