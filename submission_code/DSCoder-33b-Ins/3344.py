class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        import heapq
        points.sort()
        min_heap = []
        max_dist = 0
        for x, y in points:
            while min_heap and min_heap[0][1] < x:
                max_dist = max(max_dist, y - min_heap[0][0])
                heapq.heappop(min_heap)
            heapq.heappush(min_heap, (y, x))
        while min_heap:
            max_dist = max(max_dist, min_heap[0][0])
            heapq.heappop(min_heap)
        return max_dist