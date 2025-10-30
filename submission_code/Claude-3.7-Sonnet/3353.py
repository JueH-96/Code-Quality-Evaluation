class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        # Handle edge case
        if len(s) < 2:
            return False
        
        # Get the reverse of the string
        reversed_s = s[::-1]
        
        # Check each 2-character substring
        for i in range(len(s) - 1):
            substring = s[i:i+2]
            if substring in reversed_s:
                return True
        
        return False