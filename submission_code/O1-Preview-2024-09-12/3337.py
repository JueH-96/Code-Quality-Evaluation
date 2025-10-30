class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        positions = [i for i in range(len(s)) if s[i] == c]
        k = len(positions)
        return k * (k + 1) // 2