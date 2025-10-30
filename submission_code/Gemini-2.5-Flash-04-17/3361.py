class Solution:
    def findLatestTime(self, s: str) -> str:
        s_list = list(s)

        # Handle the hour part (HH)
        # We want the latest possible hour between 00 and 11.

        # Decide s_list[0] first, as it affects the range of s_list[1].
        if s_list[0] == '?':
            # If the second digit (s_list[1]) allows hours 10 or 11,
            # we should choose '1' for the first digit to make the hour larger.
            # This is possible if s_list[1] is '?', '0', or '1'.
            # Condition `s_list[1] <= '1'` works for '0' and '1' because of ASCII comparison.
            if s_list[1] == '?' or s_list[1] <= '1':
                s_list[0] = '1'
            else:
                # If s_list[1] is '2' through '9', we cannot form a valid hour
                # starting with '1' (12-19 are invalid). So, the first digit
                # must be '0'.
                s_list[0] = '0'

        # Decide s_list[1] if it's '?'.
        # Now decide the second digit based on the first digit (which is now known).
        if s_list[1] == '?':
            if s_list[0] == '1':
                # If the first digit is '1', the second digit can only be '0' or '1'
                # for a valid hour (10 or 11). To maximize, choose '1'.
                s_list[1] = '1'
            else: # s_list[0] must be '0' (or originally was '0')
                # If the first digit is '0', the second digit can be '0' through '9'
                # for a valid hour (00-09). To maximize, choose '9'.
                s_list[1] = '9'

        # Handle the minute part (MM)
        # We want the latest possible minute between 00 and 59.

        # First digit of the minute (s_list[3])
        if s_list[3] == '?':
            # The first digit of the minute can be '0' through '5'.
            # To maximize, choose '5'.
            s_list[3] = '5'

        # Second digit of the minute (s_list[4])
        if s_list[4] == '?':
            # The second digit of the minute can be '0' through '9'.
            # To maximize, choose '9'.
            s_list[4] = '9'

        return "".join(s_list)