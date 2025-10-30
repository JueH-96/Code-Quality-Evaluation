class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        # If s length is less than 2, no substring of length 2 exists.
        if len(s) < 2:
            return False
        
        # Compute the reverse of s.
        reversed_s = s[::-1]
        
        # Check every substring of length 2 in s.
        for i in range(len(s) - 1):
            substring = s[i:i+2]
            if substring in reversed_s:
                return True
        
        return False