class Solution:
    def scoreOfString(self, s: str) -> int:
        score = 0
        # iterate starting from the second character to the end of the string
        for i in range(1, len(s)):
            score += abs(ord(s[i]) - ord(s[i-1]))
        return score