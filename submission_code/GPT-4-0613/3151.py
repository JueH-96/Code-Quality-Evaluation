class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        processorTime.sort()
        tasks.sort(reverse=True)
        for i in range(len(processorTime)):
            for _ in range(4):
                processorTime[i] += tasks.pop()
        return max(processorTime)