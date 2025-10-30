from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        processorTime.sort()
        tasks.sort(reverse=True)
        
        max_time = 0
        task_index = 0
        
        for time in processorTime:
            max_time = max(max_time, max(time + tasks[task_index + i] for i in range(4)))
            task_index += 4
        
        return max_time