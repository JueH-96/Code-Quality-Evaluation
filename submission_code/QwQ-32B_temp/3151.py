from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        processorTime.sort()
        tasks.sort(reverse=True)
        max_time = 0
        n = len(processorTime)
        for i in range(n):
            pt = processorTime[i]
            task_idx = i * 4  # The first task in the current group of 4
            current_max = pt + tasks[task_idx]
            if current_max > max_time:
                max_time = current_max
        return max_time