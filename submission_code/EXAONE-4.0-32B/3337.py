class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        count = 0
        for char in s:
            if char == c:
                count += 1
        return count * (count + 1) // 2