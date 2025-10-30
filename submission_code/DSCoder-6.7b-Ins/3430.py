class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Sort the meetings by their starting day
        meetings.sort()

        # Initialize the count of available days and the end day of the first meeting
        available_days = days
        end_day = 0

        # Iterate over the meetings
        for start, end in meetings:
            # If the start day of the current meeting is after the end day of the previous meetings,
            # then there are available days between the end day and the start day of the current meeting
            if start > end_day:
                available_days -= start - end_day - 1
            # Update the end day of the previous meetings
            end_day = max(end_day, end)

        # Return the count of available days
        return available_days