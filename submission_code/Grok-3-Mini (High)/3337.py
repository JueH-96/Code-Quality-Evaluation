class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        count_c = s.count(c)
        return (count_c * (count_c + 1)) // 2