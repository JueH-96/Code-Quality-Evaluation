class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort the processor availability times in ascending order
        processorTime.sort()
        
        # Sort the task times in descending order
        tasks.sort(reverse=True)
        
        # We will assign the largest tasks to the processors with the smallest start times.
        # Each processor has 4 cores, so partition the tasks in groups of 4 (in descending order).
        
        max_finish_time = 0
        idx = 0
        for i, p_time in enumerate(processorTime):
            # In each iteration, assign 4 tasks to the i-th processor
            # The finish time for the processor is p_time + task_time for each assigned task
            # We keep track of the maximum among those assigned to find the processor's finish time
            for j in range(4):
                finish_time = p_time + tasks[idx]
                max_finish_time = max(max_finish_time, finish_time)
                idx += 1
        
        return max_finish_time