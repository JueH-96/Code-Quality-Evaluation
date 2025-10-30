class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort processor times in ascending order
        processorTime.sort()
        
        # Sort tasks in descending order
        tasks.sort(reverse=True)
        
        max_time = 0
        
        # For each processor, assign the next 4 largest tasks
        for i in range(len(processorTime)):
            proc_time = processorTime[i]
            
            # Get max completion time for current processor's 4 tasks
            for j in range(4):
                task_idx = i * 4 + j
                completion_time = proc_time + tasks[task_idx]
                max_time = max(max_time, completion_time)
                
        return max_time