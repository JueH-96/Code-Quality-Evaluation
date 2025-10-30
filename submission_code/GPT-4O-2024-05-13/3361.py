class Solution:
    def findLatestTime(self, s: str) -> str:
        # Convert the string to a list to manipulate characters
        time = list(s)
        
        # Handle the hour part
        if time[0] == '?':
            if time[1] == '?' or time[1] < '2':
                time[0] = '1'
            else:
                time[0] = '0'
        
        if time[1] == '?':
            if time[0] == '1':
                time[1] = '1'
            else:
                time[1] = '9'
        
        # Handle the minute part
        if time[3] == '?':
            time[3] = '5'
        
        if time[4] == '?':
            time[4] = '9'
        
        # Join the list back into a string
        return ''.join(time)