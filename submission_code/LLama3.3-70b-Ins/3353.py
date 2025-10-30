class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        # Reverse the input string
        reversed_s = s[::-1]
        
        # Iterate over all substrings of length 2 in the input string
        for i in range(len(s) - 1):
            substring = s[i:i+2]
            
            # Check if the current substring is present in the reversed string
            if substring in reversed_s:
                return True
        
        # If no matching substring is found, return False
        return False