from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort processors in ascending order of their available times
        processors_sorted = sorted(processorTime)
        # Sort tasks in descending order of their execution times
        tasks_sorted = sorted(tasks, reverse=True)
        n = len(processors_sorted)
        max_time = 0
        
        for i in range(n):
            # Each processor takes four consecutive tasks from the sorted list
            start_index = i * 4
            # The maximum task in the current group is the first one (since tasks are sorted in descending order)
            current_task = tasks_sorted[start_index]
            current_processor_time = processors_sorted[i]
            # Calculate the time taken by the current processor for its assigned tasks
            current_max = current_processor_time + current_task
            # Update the overall maximum time if the current processor's time is greater
            if current_max > max_time:
                max_time = current_max
        
        return max_time