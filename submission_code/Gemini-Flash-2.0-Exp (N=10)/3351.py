import heapq

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        heap = [-x for x in happiness]
        heapq.heapify(heap)
        ans = 0
        for _ in range(k):
            max_val = -heapq.heappop(heap)
            ans += max_val
            new_heap = []
            while heap:
                val = -heapq.heappop(heap)
                new_val = max(0, val - 1)
                heapq.heappush(new_heap, -new_val)
            heap = new_heap
        return ans