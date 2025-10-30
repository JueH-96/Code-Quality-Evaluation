class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Sort meetings by their start day
        meetings.sort(key=lambda x: x[0])
        
        total_covered = 0
        curr_start, curr_end = meetings[0]
        
        for i in range(1, len(meetings)):
            start, end = meetings[i]
            # If the current meeting interval can be merged with the previous one
            if start <= curr_end + 1:
                curr_end = max(curr_end, end)
            else:
                # Add the length of the finished interval
                total_covered += (curr_end - curr_start + 1)
                # Start a new interval
                curr_start, curr_end = start, end
        
        # Add the last interval's coverage
        total_covered += (curr_end - curr_start + 1)
        
        # The days not in any meeting is total days minus covered days
        return days - total_covered