from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort the processor times and tasks
        processorTime.sort()
        tasks.sort(reverse=True)
        
        n = len(processorTime)
        min_time = 0
        
        # Assign tasks to processors
        for i in range(n):
            # Each processor handles 4 tasks
            # We take the 4 largest remaining tasks for each processor
            max_task_time = max(tasks[i*4] + processorTime[i],
                                tasks[i*4+1] + processorTime[i],
                                tasks[i*4+2] + processorTime[i],
                                tasks[i*4+3] + processorTime[i])
            min_time = max(min_time, max_task_time)
        
        return min_time