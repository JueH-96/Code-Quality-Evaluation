class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        meetings_sorted = sorted(meetings, key=lambda x: x[0])
        merged = []
        start, end = meetings_sorted[0]
        for i in range(1, len(meetings_sorted)):
            s, e = meetings_sorted[i]
            if s <= end + 1:
                end = max(end, e)
            else:
                merged.append([start, end])
                start, end = s, e
        merged.append([start, end])
        
        total_meeting_days = 0
        for interval in merged:
            total_meeting_days += (interval[1] - interval[0] + 1)
            
        return days - total_meeting_days