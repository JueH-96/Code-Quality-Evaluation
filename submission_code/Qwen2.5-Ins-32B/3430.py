from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Sort meetings based on start time
        meetings.sort()
        
        # Initialize variables
        last_end = 0
        total_meeting_days = 0
        
        for start, end in meetings:
            # Add the duration of the current meeting to the total
            total_meeting_days += end - start + 1
            
            # Adjust for overlaps
            if start <= last_end:
                total_meeting_days -= last_end - start + 1
            
            # Update the last end time
            last_end = max(last_end, end)
        
        # Calculate available days
        available_days = days - total_meeting_days
        
        return available_days