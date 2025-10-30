class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Sort the meetings based on their start days
        meetings.sort()
        merged = [meetings[0]]
        
        for interval in meetings[1:]:
            last = merged[-1]
            if interval[0] <= last[1]:
                # Merge the intervals
                new_start = last[0]
                new_end = max(last[1], interval[1])
                merged.pop()
                merged.append([new_start, new_end])
            else:
                merged.append(interval)
        
        # Calculate the total days covered by meetings
        total_covered = 0
        for start, end in merged:
            total_covered += end - start + 1
        
        # Return the difference between total days and covered days
        return days - total_covered