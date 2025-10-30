class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort the processors by their available times
        sorted_processors = sorted(processorTime)
        # Sort the tasks in descending order
        sorted_tasks = sorted(tasks, reverse=True)
        
        max_time = 0
        for i in range(len(sorted_processors)):
            # Assign the next four tasks to this processor
            start = i * 4
            end = start + 4
            current_tasks = sorted_tasks[start:end]
            # Calculate the maximum completion time for this processor
            current_max = max(sorted_processors[i] + t for t in current_tasks)
            if current_max > max_time:
                max_time = current_max
        return max_time