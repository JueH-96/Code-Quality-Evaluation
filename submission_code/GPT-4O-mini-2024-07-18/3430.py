class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Create a set to track all the days that are occupied by meetings
        occupied_days = set()
        
        # Iterate through each meeting and mark the days as occupied
        for start, end in meetings:
            for day in range(start, end + 1):
                occupied_days.add(day)
        
        # Calculate the total available days by subtracting occupied days from total days
        available_days = days - len(occupied_days)
        
        return available_days