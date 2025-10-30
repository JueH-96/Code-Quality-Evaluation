from typing import List
import heapq

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        # max-heap of (-count, id)
        max_heap = []
        # current counts of each id
        id_count = {}
        ans = []
        
        for num, delta in zip(nums, freq):
            old_count = id_count.get(num, 0)
            new_count = old_count + delta
            
            # Update the id_count map
            if new_count == 0:
                # remove the id entirely if its count drops to zero
                id_count.pop(num, None)
            else:
                id_count[num] = new_count
                # push the new count into the heap
                heapq.heappush(max_heap, (-new_count, num))
            
            # Clean up the heap top if it's stale
            while max_heap:
                neg_cnt, top_id = max_heap[0]
                actual_cnt = id_count.get(top_id, 0)
                # if the heap entry matches the actual current count, we're good
                if actual_cnt == -neg_cnt:
                    break
                # otherwise pop stale entry
                heapq.heappop(max_heap)
            
            # Determine current max frequency
            if max_heap:
                ans.append(-max_heap[0][0])
            else:
                ans.append(0)
        
        return ans