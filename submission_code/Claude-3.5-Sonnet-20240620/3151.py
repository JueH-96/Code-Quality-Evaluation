class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort processorTime in ascending order
        processorTime.sort()
        
        # Sort tasks in descending order
        tasks.sort(reverse=True)
        
        max_time = 0
        task_index = 0
        
        # Iterate through each processor
        for proc_time in processorTime:
            # For each processor, find the maximum time among its 4 tasks
            proc_max_time = proc_time + max(tasks[task_index:task_index+4])
            # Update the overall maximum time if necessary
            max_time = max(max_time, proc_max_time)
            # Move to the next set of 4 tasks
            task_index += 4
        
        return max_time