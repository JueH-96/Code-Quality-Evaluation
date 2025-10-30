class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        for i in range(len(s)):
            for j in range(i+2, len(s)+1):
                if j-i == 2 and s[i:j] == s[i:j][::-1]:
                    return True
        return False