class Solution:
    def findLatestTime(self, s: str) -> str:
        hours, minutes = s.split(':')
        
        # Replace '?' in minutes first
        if minutes[1] == '?':
            minutes = minutes[0] + '9'
        if minutes[0] == '?':
            if minutes[1] == '9':
                minutes = '5' + '9'
            else:
                minutes = '5' + minutes[1]
        
        # Now handle hours
        if hours[1] == '?':
            if hours[0] == '1' or hours[0] == '?':
                hours = '1' + '1'  # Latest is 11
            else:
                hours = '0' + '9'  # Latest is 09
        if hours[0] == '?':
            if hours[1] == '1' or hours[1] == '0':
                hours = '0' + hours[1]  # Keep it as 01 or 00
            else:
                hours = '0' + '9'  # Latest is 09
        
        return hours + ':' + minutes