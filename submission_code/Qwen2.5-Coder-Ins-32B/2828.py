class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        i = 0
        
        # Find the first character that is not 'a'
        while i < n and s[i] == 'a':
            i += 1
        
        # If the entire string is 'a's, change the last character to 'z'
        if i == n:
            return s[:-1] + 'z'
        
        # Replace characters until we encounter an 'a'
        while i < n and s[i] != 'a':
            s = s[:i] + chr(ord(s[i]) - 1) + s[i+1:]
            i += 1
        
        return s