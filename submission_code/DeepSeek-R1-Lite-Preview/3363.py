import heapq
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        id_count = {}
        heap = []
        ans = []
        
        for i in range(len(nums)):
            num = nums[i]
            f = freq[i]
            
            if f > 0:
                id_count[num] = id_count.get(num, 0) + f
            else:
                id_count[num] = id_count.get(num, 0) - (-f)
            
            # Push the current count and ID onto the heap
            heapq.heappush(heap, (-id_count[num], num))
            
            # Remove stale entries from the heap
            while heap:
                current_max_neg, current_id = heap[0]
                if id_count[current_id] == -current_max_neg:
                    break
                heapq.heappop(heap)
            
            # The current max is the negative of the heap top's count
            current_max = -heap[0][0]
            ans.append(current_max)
        
        return ans