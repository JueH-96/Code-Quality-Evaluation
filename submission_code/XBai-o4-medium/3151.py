class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort the processor times in ascending order
        processorTime.sort()
        # Sort the tasks in descending order
        tasks.sort(reverse=True)
        max_time = 0
        n = len(processorTime)
        # Iterate over each processor
        for i in range(n):
            # The maximum task time for the current processor's group of 4 tasks
            current_time = processorTime[i] + tasks[i * 4]
            if current_time > max_time:
                max_time = current_time
        return max_time