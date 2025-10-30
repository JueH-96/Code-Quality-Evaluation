class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        processorTime.sort()
        tasks.sort(reverse=True)
        max_time = 0
        n = len(processorTime)
        for i in range(n):
            current_time = processorTime[i] + tasks[i * 4]
            if current_time > max_time:
                max_time = current_time
        return max_time