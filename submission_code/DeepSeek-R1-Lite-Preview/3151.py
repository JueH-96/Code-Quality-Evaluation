from typing import List
import heapq

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort tasks in descending order
        tasks_sorted = sorted(tasks, reverse=True)
        
        # Initialize min-heap with each processor's availability time, repeated 4 times
        heap = []
        for time in processorTime:
            for _ in range(4):
                heap.append(time)
        heapq.heapify(heap)
        
        # Initialize the maximum finish time
        max_time = 0
        
        # Assign each task to the earliest available core
        for task in tasks_sorted:
            earliest_available = heapq.heappop(heap)
            new_available = earliest_available + task
            heapq.heappush(heap, new_available)
            if new_available > max_time:
                max_time = new_available
        
        return max_time