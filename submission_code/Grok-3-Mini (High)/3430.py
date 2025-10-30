class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Sort the meetings by start day
        meetings.sort()
        
        # Initialize variables for merging intervals
        current_start = meetings[0][0]
        current_end = meetings[0][1]
        covered_days = 0
        
        # Iterate through the meetings starting from the second one
        for start, end in meetings[1:]:
            if start <= current_end:
                # Overlapping intervals, merge them
                current_end = max(current_end, end)
            else:
                # Non-overlapping, add the length of the current merged interval
                covered_days += current_end - current_start + 1
                # Start a new merged interval
                current_start = start
                current_end = end
        
        # Add the length of the last merged interval
        covered_days += current_end - current_start + 1
        
        # The result is total days minus covered days
        return days - covered_days