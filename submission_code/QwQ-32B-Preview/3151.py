from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        processorTime.sort()
        tasks.sort(reverse=True)
        n = len(processorTime)
        max_finish_time = 0
        for i in range(n):
            max_task = tasks[4 * i]
            finish_time = processorTime[i] + max_task
            if finish_time > max_finish_time:
                max_finish_time = finish_time
        return max_finish_time