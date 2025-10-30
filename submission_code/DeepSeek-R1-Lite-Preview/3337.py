class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        # Count the number of times c appears in s
        k = s.count(c)
        # Calculate the number of substrings using the formula k * (k + 1) // 2
        return k * (k + 1) // 2