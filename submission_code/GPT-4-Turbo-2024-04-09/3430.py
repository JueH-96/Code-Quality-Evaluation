class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        if not meetings:
            return days
        
        # Sort meetings based on start time
        meetings.sort()
        
        # Merge overlapping intervals
        merged_meetings = []
        current_start, current_end = meetings[0]
        
        for start, end in meetings:
            if start <= current_end:  # Overlapping or contiguous intervals
                current_end = max(current_end, end)
            else:
                merged_meetings.append([current_start, current_end])
                current_start, current_end = start, end
        
        # Append the last merged interval
        merged_meetings.append([current_start, current_end])
        
        # Calculate the number of days without meetings
        free_days = days
        
        for start, end in merged_meetings:
            free_days -= (end - start + 1)
        
        return free_days