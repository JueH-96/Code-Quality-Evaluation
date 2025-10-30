class Solution:
    def findLatestTime(self, s: str) -> str:
        # Convert string to list for easier manipulation
        time = list(s)
        
        # Handle first digit (tens place of hours)
        if time[0] == '?':
            if time[1] == '?' or int(time[1]) <= 1:
                time[0] = '1'
            else:
                time[0] = '0'
                
        # Handle second digit (ones place of hours)
        if time[1] == '?':
            if time[0] == '1':
                time[1] = '1'
            else:
                time[1] = '9'
                
        # Handle third digit (tens place of minutes)
        if time[3] == '?':
            time[3] = '5'
            
        # Handle fourth digit (ones place of minutes)
        if time[4] == '?':
            time[4] = '9'
            
        return ''.join(time)