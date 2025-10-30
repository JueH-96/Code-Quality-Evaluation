class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort processorTime in ascending order
        processorTime.sort()
        # Sort tasks in descending order
        tasks.sort(reverse=True)
        
        # We will assign the largest tasks to the earliest available processors.
        # Each processor gets exactly 4 tasks, and the finishing time for a processor
        # is its availability time + the largest task of the 4 (since they run in parallel).
        
        max_time = 0
        for i in range(len(processorTime)):
            # The largest task for this group of 4 tasks is tasks[4*i]
            finish_time = processorTime[i] + tasks[4*i]
            max_time = max(max_time, finish_time)
        
        return max_time