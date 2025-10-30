class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        occupied = [0] * (days + 1)
        for meeting in meetings:
            for i in range(meeting[0], meeting[1] + 1):
                occupied[i] = 1
        return sum(1 - x for x in occupied[1:])