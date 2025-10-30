class Solution:
    def scoreOfString(self, s: str) -> int:
        score = 0
        for i in range(len(s) - 1):
            ascii_current = ord(s[i])
            ascii_next = ord(s[i+1])
            score += abs(ascii_current - ascii_next)
        return score