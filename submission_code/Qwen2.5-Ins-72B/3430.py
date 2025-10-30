from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Create a set to keep track of days with meetings
        meeting_days = set()
        
        # Add all days with meetings to the set
        for start, end in meetings:
            for day in range(start, end + 1):
                meeting_days.add(day)
        
        # Count the number of days available for work but no meetings are scheduled
        available_days = days - len(meeting_days)
        
        return available_days