from typing import List
import heapq

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        n = len(processorTime)
        tasks = [-t for t in tasks]
        heapq.heapify(tasks)
        processorTime.sort()
        for i in range(n):
            for _ in range(4):
                task = heapq.heappop(tasks)
                processorTime[i] -= task
        return -max(processorTime)