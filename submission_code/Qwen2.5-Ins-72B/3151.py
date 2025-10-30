from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort the processor times in ascending order
        processorTime.sort()
        # Sort the tasks in descending order
        tasks.sort(reverse=True)
        
        # Initialize the maximum time to 0
        max_time = 0
        
        # Assign tasks to processors
        for i in range(len(processorTime)):
            # Calculate the time taken by the current processor to finish all its tasks
            current_time = max(processorTime[i] + tasks[i * 4 + j] for j in range(4))
            # Update the maximum time
            max_time = max(max_time, current_time)
        
        return max_time