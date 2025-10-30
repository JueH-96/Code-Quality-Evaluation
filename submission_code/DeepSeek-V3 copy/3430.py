class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Sort meetings based on start day
        meetings.sort()
        
        merged = []
        for meeting in meetings:
            if not merged:
                merged.append(meeting)
            else:
                last = merged[-1]
                if meeting[0] <= last[1] + 1:
                    merged[-1] = [min(last[0], meeting[0]), max(last[1], meeting[1])]
                else:
                    merged.append(meeting)
        
        total_meeting_days = 0
        for interval in merged:
            total_meeting_days += interval[1] - interval[0] + 1
        
        return days - total_meeting_days