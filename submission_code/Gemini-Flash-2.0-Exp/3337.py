class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        count = 0
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                if s[i] == c and s[j] == c:
                    count += 1
        return count