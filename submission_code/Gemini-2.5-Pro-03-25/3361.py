class Solution:
    """
    Provides a method to find the latest possible 12-hour format time
    by replacing '?' characters in a given time string template.
    The 12-hour format is HH:MM where 00 <= HH <= 11 and 00 <= MM <= 59.
    """
    def findLatestTime(self, s: str) -> str:
        """
        Replaces '?' characters in the time string s to form the latest possible
        valid 12-hour time (HH:MM, 00:00 to 11:59).

        Args:
            s: A string representing a 12-hour time format "HH:MM" with some
               digits possibly replaced by '?'. The string has length 5 and s[2] is ':'.
               It's guaranteed that at least one valid time can be formed.

        Returns:
            The resulting string representing the latest possible valid time.
        """
        
        # Convert the input string to a list of characters for easy modification.
        # This allows us to change characters at specific indices.
        time_list = list(s)
        
        # --- Handle Hours (HH) ---
        # We want to find the largest possible valid hour HH (00 to 11).
        # We process the hour digits from left to right (H1 then H2) to maximize the value.
        
        # Determine the first hour digit (H1 = time_list[0])
        if time_list[0] == '?':
            # We need to decide if H1 can be '1' or must be '0'.
            # Check the second hour digit (H2 = time_list[1]).
            h2_char = time_list[1]
            
            # If H2 is '?' or '0' or '1', then H1 can potentially be '1'
            # (forming hours 10 or 11). To maximize the time, we choose H1 = '1'.
            if h2_char == '?' or h2_char == '0' or h2_char == '1':
                time_list[0] = '1'
            # If H2 is '2' through '9', then H1 must be '0' to keep HH <= 11
            # (forming hours 02 through 09).
            else: 
                time_list[0] = '0'
                
        # Determine the second hour digit (H2 = time_list[1])
        # This step happens after H1 is potentially determined.
        if time_list[1] == '?':
            # Check the first hour digit (H1 = time_list[0]), which is now determined.
            
            # If H1 is '0', H2 can be any digit from '0' to '9'.
            # To maximize the hour, we set H2 to '9' (forming hour 09).
            if time_list[0] == '0':
                time_list[1] = '9'
            # If H1 is '1', H2 must be '0' or '1' to keep HH <= 11.
            # To maximize the hour, we set H2 to '1' (forming hour 11).
            else: # time_list[0] == '1'
                time_list[1] = '1'
                
        # --- Handle Minutes (MM) ---
        # We want to find the largest possible valid minute MM (00 to 59).
        # We process the minute digits from left to right (M1 then M2) to maximize the value.

        # Determine the first minute digit (M1 = time_list[3])
        if time_list[3] == '?':
            # The first minute digit M1 must be between '0' and '5' (inclusive)
            # for the minute MM to be valid (00-59).
            # To maximize the minute, we set M1 to the largest possible value, '5'.
            time_list[3] = '5'
            
        # Determine the second minute digit (M2 = time_list[4])
        # This step happens after M1 is potentially determined.
        if time_list[4] == '?':
            # The second minute digit M2 can be any digit from '0' to '9'.
            # The value of M1 (whether given or determined) doesn't further restrict M2
            # because the maximum MM we can form is 59, which is valid.
            # To maximize the minute, we set M2 to the largest possible value, '9'.
            time_list[4] = '9'
            
        # Join the modified list of characters back into a string format "HH:MM".
        return "".join(time_list)