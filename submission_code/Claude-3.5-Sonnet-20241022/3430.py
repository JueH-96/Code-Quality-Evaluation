class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Create array to track meeting days
        timeline = []
        
        # Add start and end points of meetings
        for start, end in meetings:
            timeline.append((start, 1))  # 1 for start
            timeline.append((end + 1, -1))  # -1 for end
            
        # Sort timeline by day
        timeline.sort()
        
        # Count days with no meetings
        free_days = days
        active_meetings = 0
        curr_day = 1
        
        for day, change in timeline:
            # Count days with no meetings up to current point
            if active_meetings == 0:
                free_days -= day - curr_day
                
            active_meetings += change
            curr_day = day
            
        # Count remaining days if any
        if active_meetings == 0 and curr_day <= days:
            free_days -= days - curr_day + 1
            
        return free_days