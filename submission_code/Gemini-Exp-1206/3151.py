class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        processorTime.sort()
        tasks.sort()
        ans = 0
        for i in range(len(processorTime)):
            ans = max(ans, processorTime[i] + tasks[len(tasks) - 4 * (i + 1) + 3])
        return ans