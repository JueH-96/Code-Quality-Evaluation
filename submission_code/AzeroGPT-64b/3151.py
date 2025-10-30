class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        maxTime = 0
        for p, time in enumerate(sorted(processorTime)):
            for i in range(4):
                maxTime = max(maxTime, time + sorted(tasks)[(3 * (p + 1) - 1 - i)])
        return maxTime