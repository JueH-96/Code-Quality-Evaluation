class Solution:
    def countSubstrings(self, s: str) -> int:
        count = 0
        for i in range(len(s)):
            for j in range(i, len(s)):
                substring = s[i:j+1]
                if int(substring) % int(substring[-1]) == 0 and int(substring[-1]) != 0:
                    count += 1
        return count