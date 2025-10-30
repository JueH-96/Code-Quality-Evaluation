from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        dp = [[0] * 26 for _ in range(26)]
        dp[ord(words[0][0]) - ord('a')][ord(words[0][-1]) - ord('a')] = len(words[0])
        
        for i in range(1, n):
            new_dp = [[float('inf')] * 26 for _ in range(26)]
            for start in range(26):
                for end in range(26):
                    if dp[start][end] == float('inf'):
                        continue
                    word = words[i]
                    new_start, new_end = ord(word[0]) - ord('a'), ord(word[-1]) - ord('a')
                    # Join current word to the end of the previous string
                    new_dp[start][new_end] = min(new_dp[start][new_end], dp[start][end] + len(word) - (end == new_start))
                    # Join current word to the start of the previous string
                    new_dp[new_start][end] = min(new_dp[new_start][end], dp[start][end] + len(word) - (new_end == start))
            dp = new_dp
        
        return min(dp[start][end] for start in range(26) for end in range(26) if dp[start][end] != float('inf'))