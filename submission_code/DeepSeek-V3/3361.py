class Solution:
    def findLatestTime(self, s: str) -> str:
        time = list(s)
        # Handle the hour part
        if time[0] == '?':
            if time[1] == '?':
                time[0] = '1'
                time[1] = '1'
            else:
                if time[1] > '1':
                    time[0] = '0'
                else:
                    time[0] = '1'
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
        return ''.join(time)