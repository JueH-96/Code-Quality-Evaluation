class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Create a set to store all the days when meetings are scheduled
        meeting_days = set()
        
        # Add all the days of each meeting to the set
        for start, end in meetings:
            meeting_days.update(range(start, end + 1))
        
        # Calculate the number of days without meetings
        return days - len(meeting_days)