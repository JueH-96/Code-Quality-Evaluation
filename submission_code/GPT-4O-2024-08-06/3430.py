class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Create a list to track the days with meetings
        has_meeting = [False] * (days + 1)
        
        # Mark the days that have meetings
        for start, end in meetings:
            for day in range(start, end + 1):
                has_meeting[day] = True
        
        # Count the days without meetings
        no_meeting_days = 0
        for day in range(1, days + 1):
            if not has_meeting[day]:
                no_meeting_days += 1
        
        return no_meeting_days