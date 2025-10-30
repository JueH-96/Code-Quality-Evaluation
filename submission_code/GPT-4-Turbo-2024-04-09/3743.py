class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        import heapq
        
        n = len(startTime)
        if n == 0:
            return eventTime
        
        # Calculate the initial gaps between meetings
        gaps = []
        for i in range(1, n):
            gaps.append(startTime[i] - endTime[i-1])
        
        # Include the gaps before the first meeting and after the last meeting
        initial_free = startTime[0] if startTime[0] > 0 else 0
        final_free = eventTime - endTime[-1] if endTime[-1] < eventTime else 0
        
        # Max heap to keep the largest k gaps
        max_heap = []
        
        # We use negative values because Python has a min-heap by default
        for gap in gaps:
            heapq.heappush(max_heap, -gap)
            if len(max_heap) > k:
                heapq.heappop(max_heap)
        
        # Calculate the maximum free time by considering the sum of the largest k gaps
        max_free_time = initial_free + final_free
        while max_heap:
            max_free_time += -heapq.heappop(max_heap)
        
        return max_free_time

# Example usage:
# sol = Solution()
# print(sol.maxFreeTime(10, 1, [0,2,9], [1,4,10]))  # Output: 6