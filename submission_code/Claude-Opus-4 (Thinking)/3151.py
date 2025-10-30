class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort processors by availability time (ascending)
        processorTime.sort()
        
        # Sort tasks by duration (descending)
        tasks.sort(reverse=True)
        
        max_time = 0
        n = len(processorTime)
        
        # Assign 4 tasks to each processor
        for i in range(n):
            # The longest task assigned to processor i is tasks[i * 4]
            # (since tasks are sorted in descending order)
            finish_time = processorTime[i] + tasks[i * 4]
            
            # Update the overall maximum time
            max_time = max(max_time, finish_time)
        
        return max_time