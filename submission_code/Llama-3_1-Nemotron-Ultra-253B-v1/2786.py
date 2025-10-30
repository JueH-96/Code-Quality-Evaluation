class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_length = 0
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                count = 0
                valid = True
                for k in range(i, j):
                    if s[k] == s[k + 1]:
                        count += 1
                        if count > 1:
                            valid = False
                            break
                if valid:
                    current_length = j - i + 1
                    if current_length > max_length:
                        max_length = current_length
        return max_length