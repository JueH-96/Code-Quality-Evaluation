class Solution:
    def minChanges(self, s: str) -> int:
        total = 0
        for i in range(0, len(s), 2):
            substr = s[i:i+2]
            zeros = substr.count('0')
            ones = 2 - zeros
            total += min(zeros, ones)
        return total