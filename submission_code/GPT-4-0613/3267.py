class Solution:
    def maximumLength(self, s: str) -> int:
        max_length = -1
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                substring = s[i:j]
                if len(set(substring)) == 1 and s.count(substring) >= 3:
                    max_length = max(max_length, len(substring))
        return max_length