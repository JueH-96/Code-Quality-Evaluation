class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        dp = [[float('inf')] * 26 for _ in range(n)]
        
        # Initial state
        dp[0][ord(words[0][0]) - ord('a')] = len(words[0])
        
        for i in range(1, n):
            for j in range(26):
                if dp[i-1][j] != float('inf'):
                    # Join str_i-1 with words[i]
                    if chr(j + 'a') == words[i][0]:
                        dp[i][ord(words[i][0]) - ord('a')] = min(dp[i][ord(words[i][0]) - ord('a')], dp[i-1][j] + len(words[i]) - 1)
                    else:
                        dp[i][ord(words[i][0]) - ord('a')] = min(dp[i][ord(words[i][0]) - ord('a')], dp[i-1][j] + len(words[i]))
                    
                    # Join words[i] with str_i-1
                    if words[i][-1] == chr(j + 'a'):
                        dp[i][ord(words[i][-1]) - ord('a')] = min(dp[i][ord(words[i][-1]) - ord('a')], dp[i-1][j] + len(words[i]) - 1)
                    else:
                        dp[i][ord(words[i][-1]) - ord('a')] = min(dp[i][ord(words[i][-1]) - ord('a')], dp[i-1][j] + len(words[i]))
        
        return min(dp[n-1])