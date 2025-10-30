class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort processors by availability time (ascending)
        processorTime.sort()
        
        # Sort tasks by duration (descending)
        tasks.sort(reverse=True)
        
        max_time = 0
        
        # Assign tasks to processors
        for i in range(len(processorTime)):
            # Each processor gets 4 tasks
            # The completion time for this processor is its availability time
            # plus the maximum task duration assigned to it
            # Since tasks are sorted in descending order, the first task
            # assigned to this processor is the longest
            processor_completion = processorTime[i] + tasks[i * 4]
            max_time = max(max_time, processor_completion)
        
        return max_time