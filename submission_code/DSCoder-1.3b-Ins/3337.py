class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        count = 0
        for i in range(len(s)):
            for j in range(i, len(s)):
                if s[i] == c and s[j] == c:
                    count += 1
        return count