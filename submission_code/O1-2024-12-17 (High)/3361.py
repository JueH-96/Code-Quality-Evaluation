class Solution:
    def findLatestTime(self, s: str) -> str:
        time = list(s)  # Convert to list for easy manipulation
        
        # time[0] = hour tens, time[1] = hour ones
        # time[3] = minute tens, time[4] = minute ones
        
        # 1) Fill hour tens (time[0])
        if time[0] == '?':
            # If hour ones is unknown or can be 0/1, choose '1' to get a possible hour up to '11'
            # Otherwise choose '0'
            if time[1] == '?' or time[1] in '01':
                time[0] = '1'
            else:
                time[0] = '0'
                
        # 2) Fill hour ones (time[1])
        if time[1] == '?':
            if time[0] == '0':
                time[1] = '9'  # Can go up to 09
            else:
                time[1] = '1'  # If hour tens is '1', we can only go up to 11
        
        # 3) Fill minute tens (time[3])
        if time[3] == '?':
            time[3] = '5'  # Highest possible tens for minutes (0-5)
        
        # 4) Fill minute ones (time[4])
        if time[4] == '?':
            time[4] = '9'  # Highest possible ones for minutes (0-9)
        
        return ''.join(time)