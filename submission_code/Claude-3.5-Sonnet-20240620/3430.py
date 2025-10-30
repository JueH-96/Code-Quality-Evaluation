class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        timeline = [0] * (days + 2)  # +2 for easier handling of boundaries
        
        for start, end in meetings:
            timeline[start] += 1
            timeline[end + 1] -= 1
        
        free_days = 0
        current = 0
        
        for day in range(1, days + 1):
            current += timeline[day]
            if current == 0:
                free_days += 1
        
        return free_days