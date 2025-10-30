from typing import List
import heapq

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        counts = {}
        heap = []
        ans = []
        
        for i in range(len(nums)):
            id = nums[i]
            delta = freq[i]
            
            current_count = counts.get(id, 0) + delta
            if current_count == 0:
                if id in counts:
                    del counts[id]
            else:
                counts[id] = current_count
            
            heapq.heappush(heap, (-current_count, id))
            
            current_max = 0
            while heap:
                neg_count, current_id = heap[0]
                stored_count = -neg_count
                actual_count = counts.get(current_id, 0)
                if stored_count == actual_count:
                    current_max = stored_count
                    break
                else:
                    heapq.heappop(heap)
            else:
                current_max = 0  # Executed if heap is empty
            
            ans.append(current_max)
        
        return ans