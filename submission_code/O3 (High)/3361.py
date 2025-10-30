class Solution:
    def findLatestTime(self, s: str) -> str:
        # Convert to list for easy mutation
        t = list(s)
        
        # ----- deal with hours -----
        # first hour digit (t[0])
        if t[0] == '?':
            # we would like '1' if it can lead to a valid hour
            if t[1] in ('?', '0', '1'):
                t[0] = '1'
            else:
                t[0] = '0'
                
        # second hour digit (t[1])
        if t[1] == '?':
            if t[0] == '1':          # only 0 or 1 are legal here
                t[1] = '1'
            else:                    # t[0] == '0'
                t[1] = '9'
        
        # ----- deal with minutes -----
        # minute‐tens (t[3])
        if t[3] == '?':
            t[3] = '5'               # highest legal ten-minutes digit
        
        # minute-ones (t[4])
        if t[4] == '?':
            t[4] = '9'               # always legal
        
        return ''.join(t)