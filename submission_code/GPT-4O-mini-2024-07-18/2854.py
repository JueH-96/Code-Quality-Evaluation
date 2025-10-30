class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        dp = [[float('inf')] * 2 for _ in range(n)]
        
        # Initialize the first word
        dp[0][0] = len(words[0])  # str_0 = words[0]
        dp[0][1] = len(words[0])  # str_0 = words[0]
        
        for i in range(1, n):
            word_len = len(words[i])
            for j in range(2):
                # Previous string length
                prev_length = dp[i - 1][j]
                
                # Join str_(i-1) + words[i]
                if words[i - 1][-1] == words[i][0]:
                    dp[i][0] = min(dp[i][0], prev_length + word_len - 1)
                else:
                    dp[i][0] = min(dp[i][0], prev_length + word_len)
                
                # Join words[i] + str_(i-1)
                if words[i][0] == words[i - 1][-1]:
                    dp[i][1] = min(dp[i][1], prev_length + word_len - 1)
                else:
                    dp[i][1] = min(dp[i][1], prev_length + word_len)
        
        return min(dp[n - 1])