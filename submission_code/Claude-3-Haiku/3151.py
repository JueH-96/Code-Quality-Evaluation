class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        n = len(processorTime)
        
        # Sort the tasks in descending order
        tasks.sort(reverse=True)
        
        # Initialize the available time for each processor
        available_time = [0] * n
        
        # Assign tasks to processors
        for task in tasks:
            # Find the processor with the earliest available time
            min_time_idx = available_time.index(min(available_time))
            
            # Assign the task to the processor and update its available time
            available_time[min_time_idx] += task
        
        # Return the maximum available time among all processors
        return max(available_time)