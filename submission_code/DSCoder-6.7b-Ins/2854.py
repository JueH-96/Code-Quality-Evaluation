class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        dp = [[0]*n for _ in range(n)]
        dp[0][0] = len(words[0])
        c = [[0]*26 for _ in range(26)]
        for i in range(1, n):
            c[ord(words[i-1][0])-97][ord(words[i][0])-97] = c[ord(words[i][0])-97][ord(words[i-1][0])-97] = 1
        for l in range(2, n+1):
            for i in range(n-l+1):
                j = i+l-1
                if l == 2:
                    dp[i][j] = len(words[i])+len(words[j])-c[ord(words[i][-1])-97][ord(words[j][0])-97]
                else:
                    dp[i][j] = min(dp[i+1][j]+len(words[i])-c[ord(words[i][-1])-97][ord(words[i+1][0])-97, dp[i][j-1]+len(words[j])-c[ord(words[j-1][-1])-97][ord(words[j][0])-97]])
        return dp[0][n-1]