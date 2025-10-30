from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # If there are no meetings, all days are available
        if not meetings:
            return days

        # Sort meetings by start day
        meetings.sort(key=lambda x: x[0])

        total_covered = 0
        cur_start, cur_end = meetings[0]

        # Merge overlapping intervals and accumulate covered days
        for s, e in meetings[1:]:
            if s > cur_end:
                # Disjoint interval: add the previous interval's coverage
                total_covered += (cur_end - cur_start + 1)
                cur_start, cur_end = s, e
            else:
                # Overlapping or adjacent: extend the current interval
                cur_end = max(cur_end, e)

        # Add the last interval's coverage
        total_covered += (cur_end - cur_start + 1)

        # Available days are total days minus covered days
        return days - total_covered