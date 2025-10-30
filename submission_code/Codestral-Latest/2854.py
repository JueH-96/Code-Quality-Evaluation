class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        if n == 1:
            return len(words[0])

        # Initialize the dp table
        dp = [[0] * 2 for _ in range(n)]

        # Base case
        dp[0][0] = len(words[0])
        dp[0][1] = len(words[0])

        for i in range(1, n):
            word = words[i]
            len_word = len(word)

            # Case 1: join(str_i-1, words[i])
            dp[i][0] = min(dp[i-1][0] + len_word, dp[i-1][1] + len_word)
            if words[i-1][-1] == word[0]:
                dp[i][0] -= 1

            # Case 2: join(words[i], str_i-1)
            dp[i][1] = min(dp[i-1][0] + len_word, dp[i-1][1] + len_word)
            if word[-1] == words[i-1][0]:
                dp[i][1] -= 1

        return min(dp[n-1][0], dp[n-1][1])