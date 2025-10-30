from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        if not meetings:
            return days
        # Sort meetings based on their start time
        meetings.sort()
        merged = [list(meetings[0])]
        for s, e in meetings[1:]:
            last_s, last_e = merged[-1]
            if s <= last_e:
                # Merge the intervals
                merged[-1][1] = max(last_e, e)
            else:
                merged.append([s, e])
        # Calculate total covered days
        covered = 0
        for s, e in merged:
            covered += e - s + 1
        return days - covered