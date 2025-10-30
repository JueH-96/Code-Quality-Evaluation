import heapq

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        heap = [1]
        visited = set([1])
        for _ in range(k):
            curr = heapq.heappop(heap)
            for coin in coins:
                new = curr * coin
                if new not in visited:
                    visited.add(new)
                    heapq.heappush(heap, new)
        return curr