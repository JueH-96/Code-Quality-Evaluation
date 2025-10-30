from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        """
        Calculates the number of days an employee is available for work,
        given the total number of days and a list of scheduled meetings.
        """
        
        # If there are no meetings, all days are free.
        if not meetings:
            return days

        # Sort the meetings based on their start day. This is crucial for
        # processing intervals in chronological order.
        meetings.sort()

        free_days = 0
        
        # last_covered_day tracks the end of the last processed meeting block.
        # We start at day 0, as the first work day is 1.
        last_covered_day = 0
        
        # Iterate through the sorted meetings to find gaps between them.
        for start, end in meetings:
            # A gap of free days exists if the current meeting starts after
            # the last covered day. The gap is from `last_covered_day + 1`
            # to `start - 1`.
            gap = start - last_covered_day - 1
            if gap > 0:
                free_days += gap
            
            # Update the last covered day to the end of the current merged interval.
            # This effectively merges overlapping or adjacent meetings.
            last_covered_day = max(last_covered_day, end)

        # After iterating through all meetings, account for any free days
        # between the end of the last meeting block and the total number of days.
        if days > last_covered_day:
            free_days += days - last_covered_day
            
        return free_days