class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        chars = list(s)
        
        # Find the first non-'a' character
        start = 0
        while start < n and chars[start] == 'a':
            start += 1
        
        # If all characters are 'a', we must change the last one
        if start == n:
            chars[-1] = 'z'
            return ''.join(chars)
        
        # Transform characters starting from the first non-'a' until we hit an 'a'
        i = start
        while i < n and chars[i] != 'a':
            if chars[i] == 'a':
                chars[i] = 'z'
            else:
                chars[i] = chr(ord(chars[i]) - 1)
            i += 1
        
        return ''.join(chars)