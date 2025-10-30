from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        if not meetings:
            return days
        
        # Sort meetings by start day
        sorted_meetings = sorted(meetings, key=lambda x: x[0])
        
        # Merge overlapping intervals
        merged = []
        current_start, current_end = sorted_meetings[0]
        for start, end in sorted_meetings[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))
        
        # Calculate total covered days
        covered_days = 0
        for start, end in merged:
            covered_days += min(end, days) - start + 1
            if end >= days:
                break  # No need to process further meetings
        
        # Calculate free days
        free_days = days - covered_days
        return free_days