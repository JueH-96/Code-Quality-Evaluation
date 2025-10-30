class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        dp = [[0] * 26 for _ in range(26)]
        dp[ord(words[0][0]) - ord('a')][ord(words[0][-1]) - ord('a')] = len(words[0])
        
        for i in range(1, n):
            new_dp = [[0] * 26 for _ in range(26)]
            for start, start_char in enumerate(dp):
                for end, end_char in enumerate(start_char):
                    if words[i][0] == chr(end + ord('a')):
                        new_dp[ord(words[i][0]) - ord('a')][end] = max(new_dp[ord(words[i][0]) - ord('a')][end], end_char + len(words[i]) - 1)
                    else:
                        new_dp[ord(words[i][0]) - ord('a')][end] = max(new_dp[ord(words[i][0]) - ord('a')][end], end_char + len(words[i]))
                    if words[i][-1] == chr(start + ord('a')):
                        new_dp[start][ord(words[i][-1]) - ord('a')] = max(new_dp[start][ord(words[i][-1]) - ord('a')], end_char + len(words[i]) - 1)
                    else:
                        new_dp[start][ord(words[i][-1]) - ord('a')] = max(new_dp[start][ord(words[i][-1]) - ord('a')], end_char + len(words[i]))
            dp = new_dp
        
        return max(dp)