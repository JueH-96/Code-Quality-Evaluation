class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set("aeiou")
        consonants = set("bcdfghjklmnpqrstvwxyz")
        word = '^' + word + '$'
        dp = [[0]*27 for _ in range(len(word))]
        dp[0][26] = 1
        res = sum(dp[i][25] for i in range(1, len(word)))
        for i in range(1, len(word)):
            for j in range(26):
                dp[i][j] = dp[i-1][j]
            if word[i] in vowels:
                dp[i][26] = dp[i-1][26] + 1
            if word[i] in consonants:
                dp[i][ord(word[i])-ord('a')] = dp[i-1][ord(word[i])-ord('a')] + 1
            if dp[i][26] >= k and dp[i][ord(word[i])-ord('a')] == k:
                res += 1
        return res