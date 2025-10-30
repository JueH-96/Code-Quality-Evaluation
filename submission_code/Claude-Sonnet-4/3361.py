class Solution:
    def findLatestTime(self, s: str) -> str:
        result = list(s)
        
        # Handle first digit of hour (position 0)
        if result[0] == '?':
            result[0] = '1'  # Maximum possible for first digit of hour
        
        # Handle second digit of hour (position 1)
        if result[1] == '?':
            if result[0] == '0':
                result[1] = '9'  # If hour starts with 0, second digit can be up to 9
            else:  # result[0] == '1'
                result[1] = '1'  # If hour starts with 1, second digit can only be up to 1
        
        # Handle first digit of minute (position 3)
        if result[3] == '?':
            result[3] = '5'  # Maximum possible for first digit of minute
        
        # Handle second digit of minute (position 4)
        if result[4] == '?':
            result[4] = '9'  # Maximum possible for second digit of minute
        
        return ''.join(result)