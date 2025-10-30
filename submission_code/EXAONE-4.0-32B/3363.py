import heapq
from collections import defaultdict
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        count = defaultdict(int)
        bucket = {}
        heap = []
        ans = [0] * len(nums)
        
        for i in range(len(nums)):
            id_val = nums[i]
            c = freq[i]
            old = count[id_val]
            new = old + c
            
            if old > 0:
                bucket[old] -= 1
                if bucket[old] == 0:
                    del bucket[old]
            
            count[id_val] = new
            
            if new > 0:
                if new in bucket:
                    bucket[new] += 1
                else:
                    bucket[new] = 1
                heapq.heappush(heap, -new)
            
            current_max = 0
            while heap:
                candidate = -heap[0]
                if candidate in bucket:
                    current_max = candidate
                    break
                else:
                    heapq.heappop(heap)
            
            ans[i] = current_max
        
        return ans