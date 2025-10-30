class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        count = s.count(c)  # count occurrences of c in s
        return count * (count + 1) // 2