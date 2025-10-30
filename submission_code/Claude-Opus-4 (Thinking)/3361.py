class Solution:
    def findLatestTime(self, s: str) -> str:
        result = list(s)
        
        # Handle hour
        if result[0] == '?' and result[1] == '?':
            # Both hour digits are '?', set to "11"
            result[0] = '1'
            result[1] = '1'
        elif result[0] == '?':
            # Only first hour digit is '?'
            if result[1] <= '1':
                result[0] = '1'
            else:
                result[0] = '0'
        elif result[1] == '?':
            # Only second hour digit is '?'
            if result[0] == '1':
                result[1] = '1'
            else:
                result[1] = '9'
        
        # Handle minutes
        if result[3] == '?':
            result[3] = '5'
        if result[4] == '?':
            result[4] = '9'
        
        return ''.join(result)