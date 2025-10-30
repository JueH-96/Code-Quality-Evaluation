class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        if not meetings:
            return days
        
        meetings.sort(key=lambda x: x[0])
        merged = [meetings[0]]
        
        for i in range(1, len(meetings)):
            current_meeting = meetings[i]
            last_meeting = merged[-1]
            if current_meeting[0] <= last_meeting[1] + 1:
                merged[-1][1] = max(last_meeting[1], current_meeting[1])
            else:
                merged.append(current_meeting)
        
        total_covered = 0
        for interval in merged:
            total_covered += (interval[1] - interval[0] + 1)
        
        return days - total_covered