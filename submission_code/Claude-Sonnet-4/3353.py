class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        if len(s) < 2:
            return False
        
        reverse_s = s[::-1]
        
        # Check all substrings of length 2
        for i in range(len(s) - 1):
            substring = s[i:i+2]
            if substring in reverse_s:
                return True
        
        return False