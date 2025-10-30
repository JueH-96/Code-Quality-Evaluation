class Solution:
    def findLatestTime(self, s: str) -> str:
        s = list(s)
        
        # Replace hour's first digit
        if s[0] == '?':
            s[0] = '1' if s[1] == '?' or s[1] <= '1' else '0'
        
        # Replace hour's second digit
        if s[1] == '?':
            if s[0] == '1':
                s[1] = '1'
            else:
                s[1] = '9'
        
        # Replace minute's first digit
        if s[3] == '?':
            s[3] = '5'
        
        # Replace minute's second digit
        if s[4] == '?':
            s[4] = '9'
        
        return ''.join(s)