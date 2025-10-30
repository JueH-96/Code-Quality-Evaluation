from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort the processor times in ascending order
        processorTime.sort()
        # Sort the tasks in descending order
        tasks.sort(reverse=True)
        
        max_time = 0
        n = len(processorTime)
        
        for i in range(n):
            # The start index of the current group of 4 tasks
            task_index = i * 4
            # Calculate the maximum completion time for the current processor
            current_max = processorTime[i] + tasks[task_index]
            # Update the overall maximum completion time
            if current_max > max_time:
                max_time = current_max
        
        return max_time