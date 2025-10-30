class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        processorTime.sort()
        tasks.sort(reverse=True)
        n = len(processorTime)
        task_index = 0
        max_time = 0
        for i in range(n):
            current_processor_max_time = 0
            for j in range(4):
                current_time = processorTime[i] + tasks[task_index]
                current_processor_max_time = max(current_processor_max_time, current_time)
                task_index += 1
            max_time = max(max_time, current_processor_max_time)
        return max_time