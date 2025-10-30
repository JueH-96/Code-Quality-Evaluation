from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort processorTime in ascending order
        processorTime.sort()
        # Sort tasks in descending order
        tasks.sort(reverse=True)
        
        max_time = 0
        # Assign the longest tasks to the earliest available processors
        for i in range(len(processorTime)):
            # Each processor has 4 cores, so we assign 4 tasks to each
            for j in range(4):
                current_time = processorTime[i] + tasks[i * 4 + j]
                if current_time > max_time:
                    max_time = current_time
        return max_time