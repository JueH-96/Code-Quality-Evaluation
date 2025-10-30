class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        result = list(s)
        
        # Find the first non-'a' character
        start = 0
        while start < n and s[start] == 'a':
            start += 1
        
        # If the entire string is 'a', change the last character to 'z'
        if start == n:
            result[-1] = 'z'
            return ''.join(result)
        
        # Find the end of the substring to be modified
        end = start
        while end < n and s[end] != 'a':
            end += 1
        
        # Modify the substring
        for i in range(start, end):
            result[i] = chr((ord(s[i]) - ord('a') - 1) % 26 + ord('a'))
        
        return ''.join(result)