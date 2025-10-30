class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Sort the meetings by their start time
        meetings.sort(key=lambda x: x[0])
        
        # Initialize the count of available days
        available_days = days
        
        # Iterate through the meetings
        current_end = 0
        for start, end in meetings:
            # If the current meeting starts after the previous meeting ended,
            # then the employee is available for the days between the end of
            # the previous meeting and the start of the current meeting
            if start > current_end:
                available_days -= start - current_end
            
            # Update the current end time to the maximum of the current end
            # time and the end time of the current meeting
            current_end = max(current_end, end)
        
        return available_days