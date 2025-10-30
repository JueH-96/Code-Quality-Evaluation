class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        if not meetings:
            return days
        
        # Sort meetings by their start time
        meetings.sort(key=lambda x: x[0])
        
        # Merge overlapping intervals
        merged = [meetings[0]]
        for meet in meetings[1:]:
            last_start, last_end = merged[-1]
            current_start, current_end = meet
            if current_start <= last_end:
                # Merge the intervals
                new_start = last_start
                new_end = max(last_end, current_end)
                merged[-1] = [new_start, new_end]
            else:
                merged.append(meet)
        
        # Calculate total days covered by merged intervals
        total_covered = 0
        for s, e in merged:
            total_covered += e - s + 1
        
        return days - total_covered