class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        i = 0
        while i < n and s[i] == 'a':
            i += 1
            
        if i == n:
            return s[:-1] + 'z'
        
        s_list = list(s)
        j = i
        while j < n and s_list[j] != 'a':
            s_list[j] = chr(ord(s_list[j]) - 1)
            j += 1
            
        return ''.join(s_list)