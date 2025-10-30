class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # If there are no meetings, then all days are free
        if not meetings:
            return days
        
        # Sort meetings by start day
        meetings.sort(key=lambda x: x[0])
        
        # Initialize merged interval with the first meeting
        merged_start, merged_end = meetings[0]
        total_occupied = 0
        
        for i in range(1, len(meetings)):
            start_i, end_i = meetings[i]
            # If current meeting overlaps or touches the merged interval
            if start_i <= merged_end + 1:
                # Extend the merged interval if needed
                merged_end = max(merged_end, end_i)
            else:
                # Add the length of the merged interval to total occupied days
                total_occupied += (merged_end - merged_start + 1)
                # Move to a new interval
                merged_start, merged_end = start_i, end_i
                
        # Add the last merged interval length
        total_occupied += (merged_end - merged_start + 1)
        
        # The count of free days is total days minus occupied days
        return max(0, days - total_occupied)