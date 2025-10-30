from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Sort meetings by start day
        meetings.sort(key=lambda x: x[0])
        
        merged_intervals = []
        
        for interval in meetings:
            if not merged_intervals:
                merged_intervals.append(interval)
            else:
                last_interval = merged_intervals[-1]
                # If current meeting starts at or before last_interval ends+1, they overlap or are consecutive
                if interval[0] <= last_interval[1] + 1:
                    # Merge intervals by updating the end to the maximum end.
                    last_interval[1] = max(last_interval[1], interval[1])
                else:
                    merged_intervals.append(interval)
        
        # Calculate the total days with meetings from the merged intervals.
        total_meeting_days = 0
        for start, end in merged_intervals:
            total_meeting_days += (end - start + 1)
        
        # The available days are the total days minus the days with meetings.
        return days - total_meeting_days