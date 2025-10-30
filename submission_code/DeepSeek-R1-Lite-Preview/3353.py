class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        # If the string is too short, no substrings of length 2 exist
        if len(s) < 2:
            return False
        
        # Generate all substrings of length 2 from the original string
        substrings_s = [s[i:i+2] for i in range(len(s)-1)]
        
        # Reverse the string
        s_rev = s[::-1]
        
        # Generate all substrings of length 2 from the reversed string
        substrings_rev = [s_rev[i:i+2] for i in range(len(s_rev)-1)]
        
        # Convert the reversed substrings to a set for O(1) lookups
        set_rev = set(substrings_rev)
        
        # Check if any substring from the original string exists in the reversed substrings set
        for sub in substrings_s:
            if sub in set_rev:
                return True
        
        # If no common substring found, return False
        return False