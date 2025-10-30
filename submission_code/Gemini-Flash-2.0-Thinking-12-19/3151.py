class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        processorTime.sort()
        tasks.sort(reverse=True)
        n = len(processorTime)
        max_finish_time = 0
        for i in range(n):
            processor_finish_time = 0
            for j in range(4):
                task_index = i * 4 + j
                finish_time = processorTime[i] + tasks[task_index]
                processor_finish_time = max(processor_finish_time, finish_time)
            max_finish_time = max(max_finish_time, processor_finish_time)
        return max_finish_time