class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        # Reverse the string
        reversed_s = s[::-1]
        
        # Check for any substring of length 2 in s that is also in reversed_s
        for i in range(len(s) - 1):
            substring = s[i:i+2]
            if substring in reversed_s:
                return True
        
        return False