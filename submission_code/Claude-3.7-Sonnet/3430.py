class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        if not meetings:
            return days
        
        # Sort meetings based on start day
        meetings.sort()
        
        merged = [meetings[0]]
        
        # Merge overlapping meetings
        for meeting in meetings[1:]:
            prev_start, prev_end = merged[-1]
            curr_start, curr_end = meeting
            
            if curr_start <= prev_end:  # Overlap exists
                merged[-1] = [prev_start, max(prev_end, curr_end)]
            else:
                merged.append(meeting)
        
        # Count days covered by meetings
        days_covered = 0
        for start, end in merged:
            days_covered += (end - start + 1)
        
        return days - days_covered