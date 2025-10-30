class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        
        # If all characters are 'a', we need to change the last one to 'z'
        if all(c == 'a' for c in s):
            return s[:n-1] + 'z'
        
        # Find the first non-'a' character
        i = 0
        while i < n and s[i] == 'a':
            i += 1
        
        # Try all possible substrings starting from the first non-'a' character
        min_string = s
        for j in range(i, n):
            # Perform the operation on substring s[i:j+1]
            result = s[:i]  # Keep characters before the substring
            
            for k in range(i, j+1):
                if s[k] == 'a':
                    result += 'z'
                else:
                    result += chr(ord(s[k]) - 1)
            
            result += s[j+1:]  # Keep characters after the substring
            
            if result < min_string:
                min_string = result
        
        return min_string