class Solution:
    def findLatestTime(self, s: str) -> str:
        s_list = list(s)

        # Handle s[0] (first digit of hour)
        if s_list[0] == '?':
            # To maximize the hour, we first try to make s[0] = '1' (for 10 or 11).
            # This is only possible if s[1] can be '0' or '1' (i.e., s[1] is '?', '0', or '1').
            if s_list[1] == '?' or s_list[1] in ['0', '1']:
                s_list[0] = '1'
            else:
                # If s[1] is '2' through '9', then '1x' hours (12-19) are invalid.
                # So, s[0] must be '0' to ensure a valid hour (00-09).
                s_list[0] = '0'
        # If s_list[0] is already '0' or '1', it remains as is.

        # Handle s[1] (second digit of hour)
        if s_list[1] == '?':
            # The choice for s[1] depends on the (possibly determined) value of s[0].
            if s_list[0] == '1':
                # If the first hour digit is '1', the second digit can only be '0' or '1'.
                # To get the latest time, we choose '1' (making "11").
                s_list[1] = '1'
            elif s_list[0] == '0':
                # If the first hour digit is '0', the second digit can be '0' through '9'.
                # To get the latest time, we choose '9' (making "09").
                s_list[1] = '9'
        # If s_list[1] is already a digit, it remains as is.

        # Handle s[3] (first digit of minute)
        if s_list[3] == '?':
            # The first digit of minutes can be '0' through '5'.
            # To get the latest time, we choose '5'.
            s_list[3] = '5'
        # If s_list[3] is already a digit, it remains as is.

        # Handle s[4] (second digit of minute)
        if s_list[4] == '?':
            # The second digit of minutes can be '0' through '9'.
            # To get the latest time, we choose '9'.
            s_list[4] = '9'
        # If s_list[4] is already a digit, it remains as is.

        return "".join(s_list)