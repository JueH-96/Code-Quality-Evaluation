class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        rev_s = s[::-1]
        for i in range(len(s) - 1):
            # Take each substring of length 2
            sub = s[i:i+2]
            # Check if the substring appears in the reversed string
            if sub in rev_s:
                return True
        return False