class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        indices = [i for i, char in enumerate(s) if char == c]
        m = len(indices)
        return m * (m + 1) // 2