class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        rev_s = s[::-1]
        n = len(s)
        # if string length is less than 2, no valid substring of length 2 exists.
        if n < 2:
            return False
        
        # iterate over every substring of length 2 in s
        for i in range(n - 1):
            sub = s[i:i+2]
            # check if this substring is in the reversed string
            if sub in rev_s:
                return True
        
        return False