class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        if not meetings:
            return days
        
        # Sort the meetings by their start time
        sorted_meetings = sorted(meetings, key=lambda x: x[0])
        merged = []
        
        for meeting in sorted_meetings:
            if not merged:
                merged.append(meeting)
            else:
                last_start, last_end = merged[-1]
                current_start, current_end = meeting
                if current_start <= last_end:
                    # Merge the intervals
                    new_end = max(last_end, current_end)
                    merged[-1][1] = new_end
                else:
                    merged.append(meeting)
        
        # Calculate the total covered days
        total_covered = 0
        for interval in merged:
            start, end = interval
            total_covered += end - start + 1
        
        return days - total_covered