class Solution:
    def findLatestTime(self, s: str) -> str:
        # Convert string to list for easier manipulation
        time = list(s)
        
        # Handle hour tens digit (position 0)
        if time[0] == '?':
            # Check if we can use '1' (need to ensure hour <= 11)
            if time[1] == '?' or time[1] <= '1':
                time[0] = '1'
            else:
                time[0] = '0'
        
        # Handle hour ones digit (position 1)
        if time[1] == '?':
            if time[0] == '1':
                time[1] = '1'  # Maximum is 11
            else:
                time[1] = '9'  # Maximum is 09
        
        # Handle minute tens digit (position 3)
        if time[3] == '?':
            time[3] = '5'  # Maximum is 59
        
        # Handle minute ones digit (position 4)
        if time[4] == '?':
            time[4] = '9'  # Maximum is 59
        
        return ''.join(time)