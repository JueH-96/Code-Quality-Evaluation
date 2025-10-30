class Solution:
    def findLatestTime(self, s: str) -> str:
        # Split the time into hours and minutes
        hours, minutes = s.split(':')
        
        # Replace the unknown digits with the maximum possible value
        if hours[0] == '?':
            hours = '2' if hours[1] in '0123' or hours[1] == '?' else '1' + hours[1]
        if hours[1] == '?':
            hours = hours[0] + ('3' if hours[0] == '1' else '9')
        if minutes[0] == '?':
            minutes = '5' + minutes[1]
        if minutes[1] == '?':
            minutes = minutes[0] + '9'
        
        # Combine the hours and minutes to form the latest possible time
        return hours + ':' + minutes