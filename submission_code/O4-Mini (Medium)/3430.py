from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # If there are no meetings, all days are free
        if not meetings:
            return days
        
        # Sort meetings by start day
        meetings.sort(key=lambda x: x[0])
        
        # Merge overlapping meetings
        merged = []
        current_start, current_end = meetings[0]
        for s, e in meetings[1:]:
            if s <= current_end:
                # Overlap: extend the current interval
                current_end = max(current_end, e)
            else:
                # No overlap: save the previous interval and start a new one
                merged.append((current_start, current_end))
                current_start, current_end = s, e
        # Append the last interval
        merged.append((current_start, current_end))
        
        # Sum the total covered days
        covered = sum(end - start + 1 for start, end in merged)
        
        # Free days = total days - covered days
        return days - covered