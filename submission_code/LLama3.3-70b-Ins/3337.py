class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        count = 0
        for i in range(len(s)):
            for j in range(i, len(s)):
                substring = s[i:j+1]
                if substring[0] == c and substring[-1] == c:
                    count += 1
        return count