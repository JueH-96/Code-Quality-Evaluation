class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        count = 0
        n = len(s)
        for i in range(n):
            if s[i] == c:
                count += 1
        return count * (count + 1) // 2