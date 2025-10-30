class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        pos = [i for i, char in enumerate(s) if char == c]
        total = 0
        for i in range(len(pos)):
            total += (len(pos) - i)
        return total