from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Edge: if no meetings, then all days are free
        if not meetings:
            return days
        
        # Sort intervals by start time; if same start, sort by end
        meetings.sort(key=lambda x: (x[0], x[1]))
        
        merged = []
        current_start, current_end = meetings[0]
        
        for i in range(1, len(meetings)):
            start, end = meetings[i]
            # Overlapping or touching intervals
            if start <= current_end + 1:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        # Append the last interval
        merged.append((current_start, current_end))
        
        # Calculate days covered by meetings
        covered = 0
        for start, end in merged:
            covered += (end - start + 1)
        
        # Free days are those not covered
        return days - covered