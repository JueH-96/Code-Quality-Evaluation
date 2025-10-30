class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        indices = [i for i, char in enumerate(s) if char == c]
        n = len(indices)
        return n * (n + 1) // 2