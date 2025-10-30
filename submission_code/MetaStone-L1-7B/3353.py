class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        if len(s) < 2:
            return False
        reversed_s = s[::-1]
        for i in range(len(s) - 1):
            substr = s[i:i+2]
            if substr in reversed_s:
                return True
        return False