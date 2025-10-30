class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        meetings.sort()
        available_days = 0
        last_meeting_end = 0
        
        for start, end in meetings:
            if start > last_meeting_end:
                available_days += start - last_meeting_end
                last_meeting_end = end
            elif end > last_meeting_end:
                last_meeting_end = end
        
        available_days += days - last_meeting_end
        return available_days