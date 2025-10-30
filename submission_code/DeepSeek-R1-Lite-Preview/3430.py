from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        if not meetings:
            return days  # No meetings, all days are free
        
        # Sort meetings based on start time
        meetings.sort(key=lambda x: x[0])
        
        merged = []
        for meet in meetings:
            if not merged:
                merged.append(meet)
            else:
                last_start, last_end = merged[-1]
                current_start, current_end = meet
                if current_start <= last_end + 1:
                    # Merge the intervals
                    merged[-1][1] = max(last_end, current_end)
                else:
                    merged.append(meet)
        
        total_covered = 0
        for interval in merged:
            total_covered += interval[1] - interval[0] + 1  # inclusive end
        
        free_days = days - total_covered
        return free_days