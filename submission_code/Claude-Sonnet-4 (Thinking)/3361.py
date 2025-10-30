class Solution:
    def findLatestTime(self, s: str) -> str:
        result = list(s)
        
        # Handle hour part
        if result[0] == '?':
            if result[1] == '?' or result[1] in '01':
                result[0] = '1'
            else:
                result[0] = '0'
        
        if result[1] == '?':
            if result[0] == '1':
                result[1] = '1'
            else:
                result[1] = '9'
        
        # Handle minute part
        if result[3] == '?':
            result[3] = '5'
        
        if result[4] == '?':
            result[4] = '9'
        
        return ''.join(result)