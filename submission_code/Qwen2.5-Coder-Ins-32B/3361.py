class Solution:
    def findLatestTime(self, s: str) -> str:
        s = list(s)
        
        # Determine the hour part
        if s[0] == '?':
            if s[1] in ['4', '5', '6', '7', '8', '9']:
                s[0] = '0'
            else:
                s[0] = '1'
        
        if s[1] == '?':
            if s[0] == '1':
                s[1] = '1'
            else:
                s[1] = '9'
        
        # Determine the minute part
        if s[3] == '?':
            s[3] = '5'
        
        if s[4] == '?':
            s[4] = '9'
        
        return ''.join(s)