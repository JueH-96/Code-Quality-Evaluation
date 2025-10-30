class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        # Reverse the input string
        rev_s = s[::-1]
        
        # Iterate through the string and check for substrings of length 2
        for i in range(len(s) - 1):
            substr = s[i:i+2]
            if substr in rev_s:
                return True
        
        return False