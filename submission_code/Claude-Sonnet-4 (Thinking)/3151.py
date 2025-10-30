class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort processors by start time (ascending)
        processorTime.sort()
        
        # Sort tasks by execution time (descending)
        tasks.sort(reverse=True)
        
        max_finish_time = 0
        
        # Assign 4 tasks to each processor
        for i in range(len(processorTime)):
            # The longest task assigned to this processor (first in sorted group)
            longest_task = tasks[i * 4]
            
            # The finish time for this processor
            finish_time = processorTime[i] + longest_task
            
            # Update the maximum finish time
            max_finish_time = max(max_finish_time, finish_time)
        
        return max_finish_time