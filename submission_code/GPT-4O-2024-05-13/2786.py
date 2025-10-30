class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_length = 0
        n = len(s)
        
        for i in range(n):
            consecutive_count = 0
            for j in range(i, n):
                if j > i and s[j] == s[j - 1]:
                    consecutive_count += 1
                if consecutive_count > 1:
                    break
                max_length = max(max_length, j - i + 1)
        
        return max_length