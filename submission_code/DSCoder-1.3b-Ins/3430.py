class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Initialize a list to store the meeting days
        meeting_days = [0] * (days + 1)
        
        # Iterate over the meetings
        for start, end in meetings:
            # Mark the meeting days as 1
            for day in range(start, end + 1):
                meeting_days[day] = 1
        
        # Count the number of days where no meetings are scheduled
        return sum(meeting_days)