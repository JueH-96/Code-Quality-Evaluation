class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        
        # Find the first non-'a' character
        start = 0
        while start < n and s[start] == 'a':
            start += 1
        
        # If all characters are 'a', convert the last one
        if start == n:
            return s[:-1] + 'z'
        
        # Find the end of the substring (stop at 'a' or end of string)
        end = start
        while end < n and s[end] != 'a':
            end += 1
        
        # Apply the operation
        result = list(s)
        for i in range(start, end):
            result[i] = chr(ord(result[i]) - 1)
        
        return ''.join(result)