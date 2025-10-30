class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        # Need at least two characters to form a substring of length 2
        if len(s) < 2:
            return False
        
        # Collect all substrings of length 2 in the original string
        substrings = {s[i:i+2] for i in range(len(s) - 1)}
        
        # Scan the reversed string and check whether any of its
        # length-2 substrings appear in the original set
        rev = s[::-1]
        for i in range(len(rev) - 1):
            if rev[i:i+2] in substrings:
                return True
                
        return False