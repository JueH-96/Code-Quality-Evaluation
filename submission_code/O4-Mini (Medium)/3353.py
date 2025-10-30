class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        # If the string length is less than 2, no substring of length 2 exists.
        if len(s) < 2:
            return False
        
        # Compute the reverse of s
        rev = s[::-1]
        
        # Build a set of all substrings of length 2 in the reversed string
        rev_subs = {rev[i:i+2] for i in range(len(s) - 1)}
        
        # Check if any substring of length 2 in s is in the reversed substrings
        for i in range(len(s) - 1):
            if s[i:i+2] in rev_subs:
                return True
        
        return False