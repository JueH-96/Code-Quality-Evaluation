import heapq
class Solution:
    def maximumHappinessSum(self, happiness: list[int], k: int) -> int:
        heap = []
        for i in happiness:
            heapq.heappush(heap, -i)
        sum = 0
        for i in range(k):
            max_happiness = -heapq.heappop(heap)
            sum += max_happiness
            if max_happiness > 1:
                heapq.heappush(heap, -(max_happiness - 1))
        return sum