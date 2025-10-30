import heapq
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        freq_dict = {}
        heap = []
        ans = []
        
        for i in range(len(nums)):
            id_val = nums[i]
            change = freq[i]
            current = freq_dict.get(id_val, 0)
            new = current + change
            freq_dict[id_val] = new
            
            if new != 0:
                heapq.heappush(heap, (-new, id_val))
            
            while heap:
                neg_f, top_id = heap[0]
                if freq_dict[top_id] != -neg_f:
                    heapq.heappop(heap)
                else:
                    break
            
            ans.append(-heap[0][0] if heap else 0)
        
        return ans