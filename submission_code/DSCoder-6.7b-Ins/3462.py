class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = 'aeiou'
        dp = [0] * (len(s) + 1)
        for i in range(len(s) - 1, -1, -1):
            dp[i] = dp[i + 1]
            if s[i] in vowels:
                dp[i] += 1
            if i + 2 <= len(s) and s[i:i + 2] in vowels:
                dp[i] = max(dp[i], 1 - dp[i + 2])
            if i + 3 <= len(s) and s[i:i + 3] in vowels:
                dp[i] = max(dp[i], 1 - dp[i + 3])
        return dp[0] == 1