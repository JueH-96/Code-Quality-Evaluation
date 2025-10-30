from collections import defaultdict
import heapq
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        count = defaultdict(int)
        max_heap = []
        ans = []
        
        for num, f in zip(nums, freq):
            count[num] += f
            heapq.heappush(max_heap, (-count[num], num))
            
            # Ensure the heap top is always the most frequent element
            while max_heap and -max_heap[0][0] != count[max_heap[0][1]]:
                heapq.heappop(max_heap)
            
            # If the heap is empty, the most frequent count is 0
            if not max_heap:
                ans.append(0)
            else:
                ans.append(-max_heap[0][0])
        
        return ans