class Solution:
    def smallestString(self, s: str) -> str:
        s = list(s)
        n = len(s)
        
        # Find the first non-'a' character
        i = 0
        while i < n and s[i] == 'a':
            i += 1
        
        # If all characters are 'a', reduce the last one
        if i == n:
            s[-1] = 'z'
        else:
            # Reduce consecutive non-'a' characters starting from position i
            while i < n and s[i] != 'a':
                s[i] = chr(ord(s[i]) - 1)
                i += 1
        
        return ''.join(s)