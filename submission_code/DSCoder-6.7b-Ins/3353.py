class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        for i in range(len(s)-1):
            if s[i:i+2] == s[i:i+2][::-1]:
                return True
        return False