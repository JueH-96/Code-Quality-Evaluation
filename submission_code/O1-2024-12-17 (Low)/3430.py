class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        if not meetings:
            return days  # If there are no meetings, all days are free
        
        # Sort the meetings by their start day
        meetings.sort(key=lambda x: x[0])
        
        total_covered = 0
        current_start, current_end = meetings[0]
        
        for i in range(1, len(meetings)):
            start, end = meetings[i]
            if start <= current_end + 1:
                # Overlapping or contiguous intervals; merge them
                current_end = max(current_end, end)
            else:
                # Add the length of the merged interval to total covered
                total_covered += (current_end - current_start + 1)
                # Reset to the new interval
                current_start, current_end = start, end
                
        # Add the final merged interval's coverage
        total_covered += (current_end - current_start + 1)
        
        # Days with no meetings:
        return days - total_covered