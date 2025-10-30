class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        scheduled = set()
        for start, end in meetings:
            for day in range(start, end + 1):
                scheduled.add(day)
        
        count = 0
        for day in range(1, days + 1):
            if day not in scheduled:
                count += 1
        return count