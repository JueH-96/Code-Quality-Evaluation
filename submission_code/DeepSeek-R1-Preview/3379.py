class Solution:
    def scoreOfString(self, s: str) -> int:
        total = 0
        for i in range(len(s) - 1):
            current = ord(s[i])
            next_char = ord(s[i + 1])
            total += abs(current - next_char)
        return total