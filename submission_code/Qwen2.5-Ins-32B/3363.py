from typing import List
from collections import defaultdict
import heapq

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        count = defaultdict(int)
        max_heap = []
        result = []
        max_freq = 0
        
        for num, f in zip(nums, freq):
            count[num] += f
            if count[num] > 0:
                heapq.heappush(max_heap, (-count[num], num))
            while max_heap and count[max_heap[0][1]] != -max_heap[0][0]:
                heapq.heappop(max_heap)
            max_freq = -max_heap[0][0] if max_heap else 0
            result.append(max_freq)
        
        return result