from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Create a boolean array to mark the days with meetings
        meeting_days = [False] * (days + 1)

        # Mark the days where meetings are scheduled
        for start, end in meetings:
            for day in range(start, end + 1):
                meeting_days[day] = True

        # Count the days where no meetings are scheduled
        free_days = 0
        for day in range(1, days + 1):
            if not meeting_days[day]:
                free_days += 1

        return free_days