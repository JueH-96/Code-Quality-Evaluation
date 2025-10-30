from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        n = len(processorTime)
        tasks.sort(reverse=True)
        processorTime.sort()
        
        # Initialize the time taken by each processor to finish execution of all tasks
        time_taken = [0] * n
        
        # Assign tasks to processors
        for i, task in enumerate(tasks):
            processor_index = i % n
            time_taken[processor_index] = max(processorTime[processor_index] + task, time_taken[processor_index])
        
        # Return the maximum time taken by any processor
        return max(time_taken)