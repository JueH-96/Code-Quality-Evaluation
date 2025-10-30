class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        reverse_s = s[::-1]
        substrings = {s[i:i+2] for i in range(len(s) - 1)}
        
        for substring in substrings:
            if substring in reverse_s:
                return True
        return False