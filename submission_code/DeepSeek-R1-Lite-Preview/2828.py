class Solution:
    def smallestString(self, s: str) -> str:
        # Convert string to list for easy manipulation
        chars = list(s)
        n = len(chars)
        
        # Find the first character greater than 'a'
        start = -1
        for i in range(n):
            if chars[i] > 'a':
                start = i
                break
        
        # If all characters are 'a', replace the first 'a' with 'z'
        if start == -1:
            chars[0] = 'z'
            return ''.join(chars)
        
        # Find the end of the substring to replace
        end = start
        while end < n and chars[end] > 'a':
            end += 1
        
        # Replace characters from start to end-1
        for i in range(start, end):
            if chars[i] == 'a':
                chars[i] = 'z'
            else:
                chars[i] = chr(ord(chars[i]) - 1)
        
        # Join the list back into a string and return
        return ''.join(chars)