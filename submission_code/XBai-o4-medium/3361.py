class Solution:
    def findLatestTime(self, s: str) -> str:
        time = list(s)
        # Process the hour part
        if time[0] == '?' and time[1] == '?':
            time[0] = '1'
            time[1] = '1'
        elif time[0] == '?':
            if time[1] <= '1':
                time[0] = '1'
            else:
                time[0] = '0'
        elif time[1] == '?':
            if time[0] == '0':
                time[1] = '9'
            else:
                time[1] = '1'
        # Process the minute part
        if time[3] == '?' and time[4] == '?':
            time[3] = '5'
            time[4] = '9'
        elif time[3] == '?':
            time[3] = '5'
        elif time[4] == '?':
            time[4] = '9'
        return ''.join(time)