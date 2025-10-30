class Solution:
    def smallestString(self, s: str) -> str:
        s = list(s)
        n = len(s)
        i = 0
        
        # Find the first non-'a' character
        while i < n and s[i] == 'a':
            i += 1
        
        # If all characters are 'a', change the last one to 'z'
        if i == n:
            return ''.join(s[:-1] + ['z'])
        
        # Find the end of the consecutive non-'a' substring starting at i
        j = i
        while j < n and s[j] != 'a':
            j += 1
        
        # Apply the operation to the substring [i, j-1]
        for k in range(i, j):
            s[k] = chr(ord(s[k]) - 1)
        
        return ''.join(s)