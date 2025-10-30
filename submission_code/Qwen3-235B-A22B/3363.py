from typing import List
import heapq

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        counts = dict()
        heap = []
        ans = []
        for i in range(len(nums)):
            current_id = nums[i]
            delta = freq[i]
            prev = counts.get(current_id, 0)
            new_count = prev + delta
            
            # Update counts dictionary
            if new_count == 0:
                if current_id in counts:
                    del counts[current_id]
            else:
                counts[current_id] = new_count
            
            # Push new count into heap
            heapq.heappush(heap, (-new_count, current_id))
            
            # Determine current maximum frequency
            current_max = 0
            while True:
                if not heap:
                    break  # Heap is empty, current_max remains 0
                top_neg_count, top_id = heap[0]
                stored_c = -top_neg_count
                current_c = counts.get(top_id, 0)
                
                if stored_c == current_c:
                    current_max = stored_c
                    break
                else:
                    # Remove invalid element
                    heapq.heappop(heap)
            ans.append(current_max)
        return ans