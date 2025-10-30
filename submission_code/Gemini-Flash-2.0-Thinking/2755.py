class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [0] * (n + 1)
        dictionary_set = set(dictionary)

        for i in range(1, n + 1):
            dp[i] = dp[i - 1] + 1  # Assume the last character is extra
            for word in dictionary_set:
                if i >= len(word) and s[i - len(word) : i] == word:
                    dp[i] = min(dp[i], dp[i - len(word)])

        return dp[n]