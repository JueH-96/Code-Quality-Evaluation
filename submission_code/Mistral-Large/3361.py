class Solution:
    def findLatestTime(self, s: str) -> str:
        # Convert the string to a list for easier manipulation
        time = list(s)

        # Replace '?' in the hours part
        if time[0] == '?':
            if time[1] != '?' and int(time[1]) > 3:
                time[0] = '1'
            else:
                time[0] = '2' if time[1] == '?' else '1'

        if time[1] == '?':
            if time[0] == '0' or time[0] == '1':
                time[1] = '9'
            else:
                time[1] = '3'

        # Replace '?' in the minutes part
        if time[3] == '?':
            time[3] = '5'

        if time[4] == '?':
            time[4] = '9'

        # Convert the list back to a string
        return ''.join(time)