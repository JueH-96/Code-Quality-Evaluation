class Solution:
    def findLatestTime(self, s: str) -> str:
        """
        Finds the latest valid 12-hour time by replacing '?' characters in a given pattern.
        The time format is HH:MM, where HH is 00-11 and MM is 00-59.
        """
        # Convert the string to a list of characters for easier modification.
        time_list = list(s)

        # --- Determine the latest possible hour ---

        # Process the first digit of the hour (H1 at index 0)
        if time_list[0] == '?':
            # To get the latest time, we want to make the most significant digit as large as possible.
            # The first hour digit can be '1' or '0'. We prefer '1'.
            # A '1' is possible only if the second hour digit (H2) allows the hour to be <= 11.
            # This is true if H2 is '?', '0', or '1'.
            if time_list[1] == '?' or time_list[1] <= '1':
                time_list[0] = '1'
            else:
                # If H2 is '2' or greater, H1 must be '0' to form a valid hour (e.g., 02, 03...).
                time_list[0] = '0'
        
        # Process the second digit of the hour (H2 at index 1)
        if time_list[1] == '?':
            # The value for H2 depends on the (now determined) value of H1.
            if time_list[0] == '1':
                # If the hour is "1?", the latest valid hour is "11".
                time_list[1] = '1'
            else: # time_list[0] is '0'
                # If the hour is "0?", the latest valid hour is "09".
                time_list[1] = '9'

        # --- Determine the latest possible minute ---

        # Process the first digit of the minute (M1 at index 3)
        if time_list[3] == '?':
            # To maximize the minute, M1 should be as large as possible.
            # For a minute MM to be <= 59, the largest possible M1 is '5'.
            time_list[3] = '5'

        # Process the second digit of the minute (M2 at index 4)
        if time_list[4] == '?':
            # The second digit M2 is independent of M1 and can be up to '9'.
            time_list[4] = '9'
            
        # Join the list of characters back into a string and return.
        return "".join(time_list)