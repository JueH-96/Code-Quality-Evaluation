class Solution:
    def smallestString(self, s: str) -> str:
        # Find the longest suffix that is not all 'a's
        i = len(s) - 1
        while i >= 0 and s[i] == 'a':
            i -= 1
        
        # If the entire string is 'a's, change the last character to 'z'
        if i < 0:
            return s[:-1] + 'z'
        
        # Change all characters in the suffix to their previous character
        while i >= 0:
            if s[i] == 'a':
                s = s[:i] + 'z' + s[i+1:]
            else:
                s = s[:i] + chr(ord(s[i]) - 1) + s[i+1:]
            i -= 1
        
        return s