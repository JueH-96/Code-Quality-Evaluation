class Solution:
    def findLatestTime(self, s: str) -> str:
        # Transform the string into a list for easier mutation
        time_list = list(s)

        # time_list indices: 0 1 2 3 4
        # Corresponds to:    H0 H1 : M0 M1

        # Handle hours tens (H0)
        if time_list[0] == '?':
            # If H1 is known and >= '2', setting H0 to '1' would yield 12..19 which is invalid in 12-hour format,
            # so we must set H0 to '0'.
            # Otherwise, set H0 to '1' to maximize the hour.
            if time_list[1] != '?' and time_list[1] >= '2':
                time_list[0] = '0'
            else:
                time_list[0] = '1'

        # Handle hours ones (H1)
        if time_list[1] == '?':
            # If H0 is '1', valid hours are 10 or 11, so we pick '1' to maximize.
            # If H0 is '0', hours can be 00..09, so we pick '9' to maximize.
            if time_list[0] == '1':
                time_list[1] = '1'
            else:
                time_list[1] = '9'

        # Now we have a choice of hours. Check if the chosen hour is valid (00..11).
        # If invalid (like 19, etc.), then correct the hour if needed.
        hour = int(time_list[0]) * 10 + int(time_list[1])
        if hour > 11:
            # If hour > 11, it must be that H0 = 1 and H1 = something > 1. 
            # The only fix is to reduce H1 to '1' or if H0 is 1 and H1=9 => invalid => set to '0' or '1'
            # But by the logic above, it can only happen if H1 is '9'. Correct it to '1' (11).
            time_list[1] = '1'

        # Handle minutes tens (M0)
        if time_list[3] == '?':
            # M0 can be 0..5, so we set to '5' to maximize
            time_list[3] = '5'

        # Handle minutes ones (M1)
        if time_list[4] == '?':
            # M1 can be 0..9, so we set to '9' to maximize
            time_list[4] = '9'

        return "".join(time_list)