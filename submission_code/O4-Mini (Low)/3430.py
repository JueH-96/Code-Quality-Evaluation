from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # If no meetings, all days are free
        if not meetings:
            return days
        
        # Sort meetings by start day
        meetings.sort(key=lambda x: x[0])
        
        # Merge overlapping intervals and accumulate covered days
        covered = 0
        cur_start, cur_end = meetings[0]
        
        for s, e in meetings[1:]:
            if s <= cur_end + 1:
                # Overlaps or touches the current interval: extend it if needed
                cur_end = max(cur_end, e)
            else:
                # No overlap, finalize the current interval
                covered += (cur_end - cur_start + 1)
                cur_start, cur_end = s, e
        
        # Add the last interval
        covered += (cur_end - cur_start + 1)
        
        # Free days = total days - covered days
        return days - covered