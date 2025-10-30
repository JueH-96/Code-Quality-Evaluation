class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        if len(s) < 2:
            return False
        
        reversed_s = s[::-1]
        reversed_substrings = set()
        
        # Collect all 2-character substrings of the reversed string
        for i in range(len(reversed_s) - 1):
            reversed_substrings.add(reversed_s[i:i+2])
        
        # Check each 2-character substring of the original string
        for i in range(len(s) - 1):
            substr = s[i:i+2]
            if substr in reversed_substrings:
                return True
        
        return False