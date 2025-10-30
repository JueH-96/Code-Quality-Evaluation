class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = "aeiou"
        dp = [0] * (len(s) + 1)
        vowel_count = [0] * (len(s) + 1)
        
        for i in range(len(s) - 1, -1, -1):
            vowel_count[i] = vowel_count[i + 1] + (s[i] in vowels)
            dp[i] = vowel_count[i] % 2 ^ dp[i + 1]
        
        return dp[0]