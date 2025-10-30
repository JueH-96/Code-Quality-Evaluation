class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        if not meetings:
            return days
        
        # Sort meetings by their start time
        meetings.sort()
        merged = [meetings[0]]
        
        for meet in meetings[1:]:
            last_start, last_end = merged[-1]
            curr_start, curr_end = meet
            if curr_start <= last_end:
                # Merge the intervals
                new_start = last_start
                new_end = max(last_end, curr_end)
                merged[-1] = [new_start, new_end]
            else:
                merged.append(meet)
        
        # Calculate total days covered by merged meetings
        total_covered = 0
        for interval in merged:
            total_covered += interval[1] - interval[0] + 1
        
        return days - total_covered