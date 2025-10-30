class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        if not meetings:
            return days
        
        # Sort the meetings based on their start time
        sorted_meetings = sorted(meetings, key=lambda x: x[0])
        
        # Merge overlapping or adjacent intervals
        merged = []
        for meeting in sorted_meetings:
            if not merged:
                merged.append(meeting)
            else:
                last_start, last_end = merged[-1]
                current_start, current_end = meeting
                if current_start <= last_end:
                    # Merge the intervals
                    new_start = last_start
                    new_end = max(last_end, current_end)
                    merged[-1] = [new_start, new_end]
                else:
                    merged.append(meeting)
        
        # Calculate the total days covered by meetings
        total_meeting_days = sum(end - start + 1 for start, end in merged)
        
        # Free days are the total days minus the meeting days
        free_days = days - total_meeting_days
        
        return free_days