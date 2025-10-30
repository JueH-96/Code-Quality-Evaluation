class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort the processor times in ascending order
        processorTime.sort()
        # Sort the tasks in descending order
        tasks.sort(reverse=True)
        
        max_time = 0
        # Each processor will be assigned exactly 4 tasks
        for i, start_time in enumerate(processorTime):
            # The index of the first task assigned to processor i
            offset = 4 * i
            # Compute the finishing times for the 4 tasks on this processor
            finish_times = [start_time + tasks[offset + j] for j in range(4)]
            # Track the maximum finishing time across all processors
            max_time = max(max_time, max(finish_times))
        
        return max_time