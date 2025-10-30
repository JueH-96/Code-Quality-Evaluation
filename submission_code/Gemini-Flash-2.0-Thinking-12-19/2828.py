class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        s_list = list(s)
        start_index = -1
        for i in range(n):
            if s_list[i] != 'a':
                start_index = i
                break
        
        if start_index == -1:
            s_list[-1] = 'z'
        else:
            for i in range(start_index, n):
                if s_list[i] == 'a':
                    break
                else:
                    s_list[i] = chr(ord(s_list[i]) - 1)
                    
        return "".join(s_list)