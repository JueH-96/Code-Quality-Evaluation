class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort processors by available time (ascending)
        processorTime.sort()
        
        # Sort tasks by execution time (descending)
        tasks.sort(reverse=True)
        
        max_completion_time = 0
        
        # Assign tasks to processors
        for i in range(len(processorTime)):
            processor_start_time = processorTime[i]
            
            # Each processor has 4 cores, so assign the next 4 tasks
            processor_tasks = tasks[i * 4: i * 4 + 4]
            
            # Calculate the completion time for the processor
            processor_completion_time = processor_start_time + max(processor_tasks)
            
            max_completion_time = max(max_completion_time, processor_completion_time)
        
        return max_completion_time