from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Sort the meetings by their start time
        meetings.sort()
        merged = []
        
        for m in meetings:
            if not merged:
                merged.append(m)
            else:
                last = merged[-1]
                if m[0] <= last[1]:
                    # Merge the intervals
                    new_start = last[0]
                    new_end = max(last[1], m[1])
                    merged.pop()
                    merged.append([new_start, new_end])
                else:
                    merged.append(m)
        
        # Calculate the total days covered by meetings
        total_covered = 0
        for s, e in merged:
            total_covered += e - s + 1
        
        return days - total_covered