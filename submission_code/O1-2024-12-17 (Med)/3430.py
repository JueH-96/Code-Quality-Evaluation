class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Sort the intervals by start day, then by end day
        meetings.sort(key=lambda x: (x[0], x[1]))
        
        # This will track the total number of days occupied by meetings (union of intervals)
        total_meeting_days = 0
        
        # We'll keep track of the "current" merged interval end as we merge intervals
        current_end = 0
        
        for start, end in meetings:
            if start > current_end:
                # No overlap with the previous interval
                total_meeting_days += (end - start + 1)
                current_end = end
            else:
                # Overlapping or contiguous intervals,
                # add only if this interval extends beyond current_end
                if end > current_end:
                    total_meeting_days += (end - current_end)
                    current_end = end
        
        # Return total days minus the days occupied by meetings
        return days - total_meeting_days