from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        """
        Calculates the number of days (within the range 1..days) that are not
        covered by any of the given meeting intervals.
        """
        if not meetings:
            # No meetings at all
            return days

        # Sort intervals by start day
        meetings.sort(key=lambda x: x[0])

        total_covered = 0          # total number of days occupied by meetings
        cur_start, cur_end = meetings[0]

        for start, end in meetings[1:]:
            # If the current interval overlaps with or directly touches the previous
            # one (inclusive days), we merge them.
            if start <= cur_end + 1:
                cur_end = max(cur_end, end)
            else:
                # No overlap – finalize previous interval
                total_covered += cur_end - cur_start + 1
                cur_start, cur_end = start, end

        # Add the last merged interval
        total_covered += cur_end - cur_start + 1

        # Days without meetings
        return days - total_covered