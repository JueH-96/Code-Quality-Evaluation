from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort the processor times in ascending order
        processorTime.sort()
        # Sort the tasks in descending order
        tasks.sort(reverse=True)
        
        max_time = 0
        # Assign tasks to processors
        for i in range(len(processorTime)):
            # Calculate the maximum time for the current processor
            max_time = max(max_time, processorTime[i] + tasks[i * 4])
        
        return max_time