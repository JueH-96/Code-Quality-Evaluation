class Solution:
    def findLatestTime(self, s: str) -> str:
        # Convert the string to a list of characters for easy modification
        time = list(s)
        
        # Handle hour tens place (time[0])
        if time[0] == '?':
            # If hour ones place is a digit and greater than '1', we cannot use '1' for tens
            if time[1] != '?' and int(time[1]) > 1:
                time[0] = '0'
            else:
                time[0] = '1'
        
        # Handle hour ones place (time[1])
        if time[1] == '?':
            if time[0] == '0':
                time[1] = '9'
            else:  # time[0] == '1'
                time[1] = '1'
        
        # Now ensure the formed hour is at most 11. If it's invalid, fix it:
        hour = int(time[0]) * 10 + int(time[1])
        if hour > 11:
            # If we ended up with something > 11, force hour to be 09 (largest under '10')
            time[0], time[1] = '0', '9'

        # Handle minute tens place (time[3])
        if time[3] == '?':
            time[3] = '5'
        
        # Handle minute ones place (time[4])
        if time[4] == '?':
            time[4] = '9'

        return "".join(time)