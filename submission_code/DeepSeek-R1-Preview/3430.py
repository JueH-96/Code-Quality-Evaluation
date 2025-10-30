class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Sort the meetings based on the start day
        meetings.sort(key=lambda x: x[0])
        merged = []
        
        for meeting in meetings:
            if not merged:
                merged.append(meeting)
            else:
                last_start, last_end = merged[-1]
                current_start, current_end = meeting
                # Check if current meeting starts before or at the day after the last meeting ends
                if current_start <= last_end + 1:
                    # Merge the intervals
                    new_start = last_start
                    new_end = max(last_end, current_end)
                    merged.pop()
                    merged.append([new_start, new_end])
                else:
                    merged.append(meeting)
        
        # Calculate the total covered days
        covered = 0
        for interval in merged:
            covered += interval[1] - interval[0] + 1
        
        return days - covered