class Solution:
    def findLatestTime(self, s: str) -> str:
        hours, minutes = s.split(':')
        
        # Replace the '?' in the hours
        if hours[0] == '?':
            hours = '2' if hours[1] <= '3' else '1'
        if hours[1] == '?':
            hours = hours[0] + '9'
        
        # Replace the '?' in the minutes
        if minutes[0] == '?':
            minutes = '5'
        if minutes[1] == '?':
            minutes = minutes[0] + '9'
        
        return hours + ':' + minutes