from typing import List
import heapq

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        n = len(processorTime)
        m = len(tasks)
        
        # Sort the processor times and tasks
        processorTime.sort()
        tasks.sort(reverse=True)
        
        # Create a min-heap for the processors' finish times
        finish_times = []
        
        # Initialize the heap with the processor times
        for time in processorTime:
            heapq.heappush(finish_times, time)
        
        # Assign tasks to processors
        for task in tasks:
            # Get the processor that will be free the earliest
            earliest_finish = heapq.heappop(finish_times)
            # Calculate the new finish time for this processor
            new_finish_time = earliest_finish + task
            # Push the new finish time back into the heap
            heapq.heappush(finish_times, new_finish_time)
        
        # The answer is the maximum finish time among all processors
        return max(finish_times)