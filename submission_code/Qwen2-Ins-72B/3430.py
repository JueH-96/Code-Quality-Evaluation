class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        meetings.sort()
        heap = []
        ans = 0
        for day in range(1, days + 1):
            while heap and heap[0] < day:
                heapq.heappop(heap)
            if not meetings:
                ans += 1
            elif meetings[0][0] > day:
                ans += 1
            else:
                if meetings[0][0] == day:
                    heapq.heappush(heap, meetings[0][1])
                    meetings.pop(0)
                while heap and meetings and meetings[0][0] == day:
                    heapq.heappush(heap, meetings[0][1])
                    meetings.pop(0)
        return ans