class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        n = len(processorTime)
        tasks.sort(reverse=True)
        processorTime.sort()
        processor_finish_times = [0] * n
        task_index = 0
        for i in range(n):
            core_finish_times = [processorTime[i]] * 4
            for j in range(4):
                core_finish_times[j] += tasks[task_index]
                task_index += 1
            processor_finish_times[i] = max(core_finish_times)
        return max(processor_finish_times)