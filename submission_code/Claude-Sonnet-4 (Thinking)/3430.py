class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Sort meetings by start time
        meetings.sort()
        
        # Merge overlapping meetings
        merged = []
        for start, end in meetings:
            if merged and start <= merged[-1][1]:
                # Overlapping, merge with the last interval
                merged[-1][1] = max(merged[-1][1], end)
            else:
                # No overlap, add new interval
                merged.append([start, end])
        
        # Count days covered by merged meetings
        covered_days = 0
        for start, end in merged:
            covered_days += end - start + 1
        
        # Return free days
        return days - covered_days