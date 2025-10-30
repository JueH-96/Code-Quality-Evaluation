from collections import defaultdict
from heapq import heappush, heappop
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        count = defaultdict(int)
        freq_count = defaultdict(int)
        heap = []
        ans = []
        
        for i in range(len(nums)):
            id = nums[i]
            delta = freq[i]
            old_count = count[id]
            new_count = old_count + delta
            count[id] = new_count
            
            # Update freq_count for old_count
            if old_count > 0:
                freq_count[old_count] -= 1
                if freq_count[old_count] == 0:
                    del freq_count[old_count]
            
            # Update freq_count for new_count and push to heap
            if new_count > 0:
                freq_count[new_count] += 1
                heappush(heap, -new_count)
            
            # Find current_max by cleaning up the heap
            current_max = 0
            while heap:
                candidate = -heap[0]
                if candidate in freq_count and freq_count[candidate] > 0:
                    current_max = candidate
                    break
                else:
                    heappop(heap)
            ans.append(current_max)
        
        return ans