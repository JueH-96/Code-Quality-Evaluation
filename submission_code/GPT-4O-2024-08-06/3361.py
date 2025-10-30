class Solution:
    def findLatestTime(self, s: str) -> str:
        # Split the time into hours and minutes
        hours, minutes = s.split(':')
        
        # Process the hours
        if hours[0] == '?':
            # If the second digit is between 0 and 1, the first digit can be 1
            # Otherwise, it should be 0 to form a valid hour
            if hours[1] == '?' or hours[1] in '01':
                hours = '1' + hours[1]
            else:
                hours = '0' + hours[1]
        
        if hours[1] == '?':
            # If the first digit is 1, the second digit can be 0 or 1
            # Otherwise, it can be any digit from 0 to 9
            if hours[0] == '1':
                hours = hours[0] + '1'
            else:
                hours = hours[0] + '9'
        
        # Process the minutes
        if minutes[0] == '?':
            # The first digit of minutes can be at most 5
            minutes = '5' + minutes[1]
        
        if minutes[1] == '?':
            # The second digit of minutes can be any digit from 0 to 9
            minutes = minutes[0] + '9'
        
        # Combine the processed hours and minutes
        return hours + ':' + minutes