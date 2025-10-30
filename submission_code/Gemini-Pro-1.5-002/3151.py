class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        processorTime.sort()
        tasks.sort(reverse=True)
        n = len(processorTime)
        max_time = 0
        for i in range(n):
            for j in range(4):
                max_time = max(max_time, processorTime[i] + tasks[i * 4 + j])
        return max_time