class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        dp = [[0]*26 for _ in range(n+1)]
        for i in range(n-1, -1, -1):
            dp[i][ord(target[i])-ord('a')] = i+1
            for j in range(26):
                dp[i][j] = min(dp[i+1][j], dp[dp[i+1][j]][j] if j == ord(target[i])-ord('a') else float('inf'))
        res = float('inf')
        for i in range(n):
            res = min(res, dp[0][ord(words[i])-ord('a')] + i)
        return res if res != float('inf') else -1