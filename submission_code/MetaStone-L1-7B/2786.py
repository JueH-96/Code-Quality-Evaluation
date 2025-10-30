class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_length = 1
        n = len(s)
        for i in range(n):
            current_count = 0
            for j in range(i, n):
                if j > i and s[j] == s[j-1]:
                    current_count += 1
                if current_count > 1:
                    break
                current_length = j - i + 1
                if current_length > max_length:
                    max_length = current_length
        return max_length