class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        tasks_sorted = sorted(tasks, reverse=True)
        processorTime_sorted = sorted(processorTime)
        max_time = 0
        for i in range(len(processorTime_sorted)):
            start = processorTime_sorted[i]
            group = tasks_sorted[i*4 : (i+1)*4]
            current_max = max(start + task for task in group)
            if current_max > max_time:
                max_time = current_max
        return max_time