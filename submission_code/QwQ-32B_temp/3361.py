class Solution:
    def findLatestTime(self, s: str) -> str:
        h1, h2 = s[0], s[1]
        m1, m2 = s[3], s[4]
        
        # Process hours
        if h1 == '?' and h2 == '?':
            h1, h2 = '1', '1'
        else:
            if h1 == '?':
                if h2 <= '1':
                    h1 = '1'
                else:
                    h1 = '0'
            elif h2 == '?':
                if h1 == '1':
                    h2 = '1'
                else:  # h1 is '0'
                    h2 = '9'
        
        # Process minutes
        if m1 == '?' and m2 == '?':
            m1, m2 = '5', '9'
        else:
            if m1 == '?':
                m1 = '5'
            if m2 == '?':
                m2 = '9'
        
        return f"{h1}{h2}:{m1}{m2}"