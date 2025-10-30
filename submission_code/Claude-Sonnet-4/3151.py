class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort processor times in ascending order
        processorTime.sort()
        
        # Sort tasks in descending order (longest tasks first)
        tasks.sort(reverse=True)
        
        max_completion_time = 0
        
        # Assign 4 tasks to each processor
        for i in range(len(processorTime)):
            # For processor i, assign tasks at indices 4*i, 4*i+1, 4*i+2, 4*i+3
            # Since tasks are sorted in descending order and we want to assign
            # longest tasks to earliest available processors
            processor_start_time = processorTime[i]
            
            # The completion time for this processor is start_time + max_task_time
            # Since all cores work in parallel, we take the maximum task time
            max_task_time = tasks[4 * i]  # First task is the longest among the 4
            
            completion_time = processor_start_time + max_task_time
            max_completion_time = max(max_completion_time, completion_time)
        
        return max_completion_time