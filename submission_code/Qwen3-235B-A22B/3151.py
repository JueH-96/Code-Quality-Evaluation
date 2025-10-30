from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort the processors in ascending order of their availability
        sorted_processors = sorted(processorTime)
        
        # Sort the tasks in descending order to assign the largest tasks first
        sorted_tasks = sorted(tasks, reverse=True)
        
        max_time = 0
        n = len(sorted_processors)
        
        for i in range(n):
            # The first task in each group of four is the largest in that group
            current_task = sorted_tasks[i * 4]
            current_processor_time = sorted_processors[i]
            
            current_sum = current_processor_time + current_task
            
            if current_sum > max_time:
                max_time = current_sum
        
        return max_time