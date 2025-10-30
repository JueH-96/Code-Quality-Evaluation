class Solution:
    def findLatestTime(self, s: str) -> str:
        # Split the string into hours and minutes
        hours, minutes = s.split(':')
        
        # Replace the first hour digit
        if hours[0] == '?':
            if hours[1] == '?' or int(hours[1]) <= 1:
                hours = '1' + hours[1]
            else:
                hours = '0' + hours[1]
        
        # Replace the second hour digit
        if hours[1] == '?':
            if hours[0] == '1':
                hours = hours[0] + '1'
            else:
                hours = hours[0] + '9'
        
        # Replace the first minute digit
        if minutes[0] == '?':
            minutes = '5' + minutes[1]
        
        # Replace the second minute digit
        if minutes[1] == '?':
            minutes = minutes[0] + '9'
        
        return f"{hours}:{minutes}"