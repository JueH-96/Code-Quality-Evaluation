class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        start = -1
        end = -1
        for i in range(n):
            if s[i] != 'a':
                start = i
                break
        if start == -1:
            s_list = list(s)
            s_list[n - 1] = 'z'
            return "".join(s_list)
        
        end = start
        while end < n and s[end] != 'a':
            end += 1
        
        s_list = list(s)
        for i in range(start, end):
            s_list[i] = chr(ord(s_list[i]) - 1)
        
        return "".join(s_list)