class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_len = 0
        for i in range(len(s)):
            for j in range(i, len(s)):
                substring = s[i:j+1]
                count = 0
                is_semi_repetitive = True
                for k in range(len(substring) - 1):
                    if substring[k] == substring[k+1]:
                        count += 1
                if count > 1:
                    is_semi_repetitive = False

                if is_semi_repetitive:
                    max_len = max(max_len, len(substring))
        return max_len