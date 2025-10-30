class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Create a set to store all the days that have meetings
        meeting_days = set()
        
        # Iterate through each meeting and add all the days in the meeting range to the set
        for start, end in meetings:
            for day in range(start, end + 1):
                meeting_days.add(day)
        
        # The number of days without meetings is the total days minus the number of days with meetings
        return days - len(meeting_days)