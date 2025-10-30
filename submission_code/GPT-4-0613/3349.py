class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        max_len = 0
        for i in range(len(s)):
            count = {}
            for j in range(i, len(s)):
                if s[j] in count:
                    count[s[j]] += 1
                    if count[s[j]] > 2:
                        break
                else:
                    count[s[j]] = 1
                max_len = max(max_len, j - i + 1)
        return max_len