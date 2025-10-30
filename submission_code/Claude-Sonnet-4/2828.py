class Solution:
    def smallestString(self, s: str) -> str:
        s_list = list(s)
        n = len(s)
        
        # Find the first non-'a' character
        start = 0
        while start < n and s_list[start] == 'a':
            start += 1
        
        # If all characters are 'a', we must change the last one
        if start == n:
            s_list[n-1] = 'z'
            return ''.join(s_list)
        
        # Apply operation starting from first non-'a' character
        # Continue until we hit an 'a' or reach the end
        i = start
        while i < n and s_list[i] != 'a':
            s_list[i] = chr(ord(s_list[i]) - 1)
            i += 1
        
        return ''.join(s_list)