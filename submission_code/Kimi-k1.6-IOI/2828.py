class Solution:
    def smallestString(self, s: str) -> str:
        s_list = list(s)
        n = len(s_list)
        i = 0
        
        # Find the first character that is not 'a'
        while i < n and s_list[i] == 'a':
            i += 1
        
        if i == n:
            # All characters are 'a', change the last one to 'z'
            s_list[-1] = 'z'
        else:
            # Find the end of the consecutive non-'a' characters starting at i
            j = i
            while j < n and s_list[j] != 'a':
                j += 1
            # Convert each character in the substring s_list[i:j] to previous character
            for k in range(i, j):
                s_list[k] = chr(ord(s_list[k]) - 1)
        
        return ''.join(s_list)