class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Sort meetings based on the start day
        meetings.sort(key=lambda x: x[0])
        merged = []
        
        for meet in meetings:
            if not merged:
                merged.append(meet)
            else:
                last_start, last_end = merged[-1]
                current_start, current_end = meet
                # Check if current meeting overlaps with the last merged meeting
                if current_start <= last_end:
                    # Merge them by updating the end of the last interval
                    new_end = max(last_end, current_end)
                    merged[-1][1] = new_end
                else:
                    merged.append(meet)
        
        # Calculate the total days covered by meetings
        total_meeting_days = 0
        for start, end in merged:
            total_meeting_days += end - start + 1
        
        # The result is total days minus the days covered by meetings
        return days - total_meeting_days