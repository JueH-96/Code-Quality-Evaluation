class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        def is_almost_equal(a, b):
            return a == b or abs(ord(a) - ord(b)) == 1

        n = len(word)
        dp = [0] * n

        for i in range(1, n):
            if is_almost_equal(word[i], word[i - 1]):
                dp[i] = dp[i - 1] + 1
            else:
                dp[i] = dp[i - 1]

        return dp[-1]