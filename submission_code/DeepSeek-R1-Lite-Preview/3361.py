class Solution:
    def findLatestTime(self, s: str) -> str:
        s_list = list(s)
        
        # Process hours
        h1, h2 = s_list[0], s_list[1]
        
        if h1 == '?' and h2 == '?':
            s_list[0] = '1'
            s_list[1] = '1'
        elif h1 == '?':
            if h2 in ['0', '1']:
                s_list[0] = '1'
            else:
                s_list[0] = '0'
        elif h2 == '?':
            if h1 == '0':
                s_list[1] = '9'
            elif h1 == '1':
                s_list[1] = '1'
        
        # Process minutes
        m1, m2 = s_list[3], s_list[4]
        
        if m1 == '?':
            s_list[3] = '5'
        if m2 == '?':
            s_list[4] = '9'
        
        return ''.join(s_list)