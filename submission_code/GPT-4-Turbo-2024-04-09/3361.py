class Solution:
    def findLatestTime(self, s: str) -> str:
        # Split the time into hours and minutes
        hours, minutes = s[:2], s[3:]
        
        # Replace '?' in hours
        if hours == '??':
            hours = '11'
        elif hours[0] == '?':
            if hours[1] <= '1':
                hours = '1' + hours[1]
            else:
                hours = '1' + hours[1]
        elif hours[1] == '?':
            hours = hours[0] + '9'
        
        # Replace '?' in minutes
        if minutes == '??':
            minutes = '59'
        elif minutes[0] == '?':
            minutes = '5' + minutes[1]
        elif minutes[1] == '?':
            minutes = minutes[0] + '9'
        
        # Combine the hours and minutes with ':'
        return hours + ':' + minutes