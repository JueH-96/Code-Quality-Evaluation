class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        if not meetings:
            return days
        
        # Sort meetings by start time
        meetings.sort()
        
        # Merge overlapping meetings
        merged = [meetings[0]]
        
        for i in range(1, len(meetings)):
            current_start, current_end = meetings[i]
            last_start, last_end = merged[-1]
            
            # If current meeting overlaps with the last merged meeting
            if current_start <= last_end:
                # Extend the last meeting's end time if needed
                merged[-1][1] = max(last_end, current_end)
            else:
                # No overlap, add as a new meeting
                merged.append([current_start, current_end])
        
        # Calculate total days covered by meetings
        days_with_meetings = 0
        for start, end in merged:
            days_with_meetings += end - start + 1
        
        # Return days without meetings
        return days - days_with_meetings