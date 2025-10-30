class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        if not meetings:
            return days
        
        # Sort meetings by start time
        meetings.sort()
        
        # Merge overlapping intervals
        merged = []
        for start, end in meetings:
            if not merged or merged[-1][1] < start - 1:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        
        # Count days covered by meetings
        covered_days = 0
        for start, end in merged:
            covered_days += end - start + 1
        
        return days - covered_days