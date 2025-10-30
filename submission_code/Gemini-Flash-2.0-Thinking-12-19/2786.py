class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        max_len = 0
        for i in range(n):
            for j in range(i, n):
                substring = s[i:j+1]
                consecutive_pairs = 0
                for k in range(len(substring) - 1):
                    if substring[k] == substring[k+1]:
                        consecutive_pairs += 1
                if consecutive_pairs <= 1:
                    max_len = max(max_len, len(substring))
        return max_len