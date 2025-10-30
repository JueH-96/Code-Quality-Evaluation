class Solution:
    def findLatestTime(self, s: str) -> str:
        s_list = list(s)
        if s_list[0] == '?':
            if s_list[1] == '?' or s_list[1] <= '1' or s_list[1] > '9': # consider case like "??:00" or "??:??" or "?4:00"
                s_list[0] = '1'
            else:
                s_list[0] = '0'
        if s_list[1] == '?':
            if s_list[0] == '0':
                s_list[1] = '9'
            else: # s_list[0] == '1'
                s_list[1] = '1'
        if s_list[3] == '?':
            s_list[3] = '5'
        if s_list[4] == '?':
            s_list[4] = '9'
        return "".join(s_list)