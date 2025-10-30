import heapq
from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Use a heap to keep track of the processor with the smallest time left
        heap = [-processorTime[i] for i in range(len(processorTime))]
        heapq.heapify(heap)
        
        total_time = 0
        while tasks:
            # Get the processor with the smallest time left
            curr_processor_time = -heapq.heappop(heap)
            
            # Execute the task assigned to this processor
            curr_task = tasks.pop()
            total_time += curr_processor_time + curr_task
            
            # Update the time left for the next processor
            curr_processor_time -= curr_task
            if curr_processor_time > 0:
                heapq.heappush(heap, -curr_processor_time)
        
        return total_time