class Solution:
    def findLatestTime(self, s: str) -> str:
        # Convert to list for mutability
        time = list(s)
        
        # Process hour tens s[0]
        if time[0] == '?':
            # If hour units is unknown or is '0' or '1', we can choose '1', otherwise choose '0'
            if time[1] == '?' or int(time[1]) <= 1:
                time[0] = '1'
            else:
                time[0] = '0'
                
        # Process hour units s[1]
        if time[1] == '?':
            if time[0] == '1':
                # For hours in "1?" only 0 or 1 are allowed. To maximize, choose '1'
                time[1] = '1'
            else: # time[0] == '0'
                # For hours in "0?" any digit 0-9 gives a valid hour (max "09")
                time[1] = '9'
                
        # Process minute tens s[3]
        if time[3] == '?':
            # Minutes tens digit can be 0-5
            time[3] = '5'
            
        # Process minute ones s[4]
        if time[4] == '?':
            time[4] = '9'
            
        return "".join(time)