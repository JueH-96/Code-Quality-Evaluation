from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Create a set to store the days when meetings are scheduled
        meeting_days = set()
        
        # Iterate over each meeting
        for start, end in meetings:
            # Add all days of the meeting to the set
            meeting_days.update(range(start, end + 1))
        
        # Count the number of days when the employee is available but no meetings are scheduled
        available_days = days - len(meeting_days)
        
        # Return the count of available days
        return max(available_days, 0)