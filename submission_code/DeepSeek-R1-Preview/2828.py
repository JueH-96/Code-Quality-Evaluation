class Solution:
    def smallestString(self, s: str) -> str:
        # Check if all characters are 'a'
        if all(c == 'a' for c in s):
            return s[:-1] + 'z' if len(s) > 0 else s
        
        # Find the first non 'a' character
        first_non_a = None
        for i in range(len(s)):
            if s[i] != 'a':
                first_non_a = i
                break
        
        # Find the end of the substring (j is the index after the last character to change)
        j = first_non_a
        while j < len(s) and s[j] != 'a':
            j += 1
        
        # Convert the string to a list for easy manipulation
        s_list = list(s)
        
        # Apply the operation to each character in the substring
        for k in range(first_non_a, j):
            c = s_list[k]
            if c == 'a':
                s_list[k] = 'z'
            else:
                s_list[k] = chr(ord(c) - 1)
        
        return ''.join(s_list)