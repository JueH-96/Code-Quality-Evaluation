from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        if not meetings:
            return days
        
        meetings.sort(key=lambda x: x[0])
        merged = [meetings[0]]
        
        for s, e in meetings[1:]:
            last_s, last_e = merged[-1]
            if s <= last_e:
                merged[-1][1] = max(last_e, e)
            else:
                merged.append([s, e])
        
        total_meeting_days = 0
        for s, e in merged:
            total_meeting_days += e - s + 1
        
        return days - total_meeting_days