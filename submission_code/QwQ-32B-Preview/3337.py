class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        k = 0
        for char in s:
            if char == c:
                k += 1
        return k * (k + 1) // 2