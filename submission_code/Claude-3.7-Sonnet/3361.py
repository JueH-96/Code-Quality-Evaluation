class Solution:
    def findLatestTime(self, s: str) -> str:
        # Convert to a list for easier manipulation
        time_list = list(s)
        
        # Process the hour
        if time_list[0] == "?":
            if time_list[1] == "?" or time_list[1] <= "1":
                time_list[0] = "1"  # Can use 1 if second digit allows (making 10 or 11)
            else:
                time_list[0] = "0"  # Must use 0 if second digit is >1
                
        if time_list[1] == "?":
            if time_list[0] == "1":
                time_list[1] = "1"  # If first digit is 1, can only go up to 11
            else:  # time_list[0] == "0"
                time_list[1] = "9"  # If first digit is 0, can go up to 09
                
        # Process the minute
        if time_list[3] == "?":
            time_list[3] = "5"  # First minute digit can be at most 5
            
        if time_list[4] == "?":
            time_list[4] = "9"  # Second minute digit can be at most 9
            
        return "".join(time_list)