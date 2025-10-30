class Solution:
    def scoreOfString(self, s: str) -> int:
        score = 0
        for i in range(1, len(s)):
            if s[i] == s[i-1]:
                score += 2
            else:
                score += 1
        return score