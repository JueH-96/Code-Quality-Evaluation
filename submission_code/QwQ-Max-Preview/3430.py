from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        if not meetings:
            return days
        
        # Sort the meetings by their start time
        sorted_meetings = sorted(meetings, key=lambda x: x[0])
        merged = []
        
        for start, end in sorted_meetings:
            if not merged:
                merged.append([start, end])
            else:
                last_start, last_end = merged[-1]
                if start <= last_end:
                    # Merge the intervals
                    new_end = max(last_end, end)
                    merged[-1][1] = new_end
                else:
                    merged.append([start, end])
        
        # Calculate the total covered days
        sum_covered = sum(end - start + 1 for start, end in merged)
        
        return days - sum_covered