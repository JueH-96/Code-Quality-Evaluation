class Solution:
    def scoreOfString(self, s: str) -> int:
        score = 0
        for i in range(1, s.length):
            score += abs(ord(s[i]) - ord(s[i - 1]))
        return score