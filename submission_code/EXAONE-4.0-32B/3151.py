from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        sorted_processors = sorted(processorTime)
        sorted_tasks = sorted(tasks, reverse=True)
        
        max_finish_time = 0
        n = len(sorted_processors)
        for i in range(n):
            current_finish = sorted_processors[i] + sorted_tasks[4 * i]
            if current_finish > max_finish_time:
                max_finish_time = current_finish
                
        return max_finish_time