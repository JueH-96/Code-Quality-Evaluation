class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        # Find the first character that is not 'a'
        i = 0
        while i < n and s[i] == 'a':
            i += 1
        
        # If all characters are 'a', change the last 'a' to 'z'
        if i == n:
            return s[:-1] + 'z'
        
        # Find the first 'a' after the first non-'a' character
        j = i
        while j < n and s[j] != 'a':
            j += 1
        
        # Replace characters from i to j-1
        new_s = list(s)
        for k in range(i, j):
            new_s[k] = chr(ord(s[k]) - 1)
        
        return ''.join(new_s)