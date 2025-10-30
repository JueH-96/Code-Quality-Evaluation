from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Sort meetings by their start day
        meetings.sort()
        
        # Initialize variables to keep track of the current day and available days count
        current_day = 1
        available_days = 0
        
        # Iterate through each meeting
        for start, end in meetings:
            # Count available days before the current meeting starts
            if current_day < start:
                available_days += start - current_day
            
            # Move the current day to the day after the current meeting ends
            current_day = max(current_day, end + 1)
        
        # Count available days after the last meeting ends
        if current_day <= days:
            available_days += days - current_day + 1
        
        return available_days