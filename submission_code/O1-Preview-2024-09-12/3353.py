class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        n = len(s)
        reversed_s = s[::-1]
        for i in range(n - 1):
            sub = s[i:i+2]
            if sub in reversed_s:
                return True
        return False