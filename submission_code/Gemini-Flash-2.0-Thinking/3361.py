class Solution:
    def findLatestTime(self, s: str) -> str:
        time_list = list(s)

        # Handle hours
        if time_list[0] == '?':
            if time_list[1] == '?':
                time_list[0] = '1'
            elif int(time_list[1]) <= 1:
                time_list[0] = '1'
            else:
                time_list[0] = '0'

        if time_list[1] == '?':
            if time_list[0] == '0':
                time_list[1] = '9'
            elif time_list[0] == '1':
                time_list[1] = '1'

        # Handle minutes
        if time_list[3] == '?':
            time_list[3] = '5'

        if time_list[4] == '?':
            time_list[4] = '9'

        return "".join(time_list)