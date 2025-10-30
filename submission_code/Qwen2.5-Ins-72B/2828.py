class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        s = list(s)
        
        # Find the first non-'a' character
        i = 0
        while i < n and s[i] == 'a':
            i += 1
        
        # If the entire string is 'a's, change the last 'a' to 'z'
        if i == n:
            s[-1] = 'z'
        else:
            # Decrement characters until we hit an 'a'
            while i < n and s[i] != 'a':
                s[i] = chr(ord(s[i]) - 1)
                i += 1
        
        return ''.join(s)