class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        count = s.count(c)
        if count < 2:
            return count
        return count * (count + 1) // 2