class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort the processor times in ascending order
        processorTime.sort()
        
        # Sort the task times in descending order
        tasks.sort(reverse=True)
        
        # We know there are exactly 4*n tasks for n processors (each has 4 cores),
        # so we can group tasks in blocks of 4 from largest to smallest.
        # Because tasks are in descending order, the largest in each block is tasks[4*i].
        
        max_completion_time = 0
        for i in range(len(processorTime)):
            # finishing time = processor start time + the largest task in its assigned block of 4
            completion_time = processorTime[i] + tasks[4*i]
            max_completion_time = max(max_completion_time, completion_time)
        
        return max_completion_time