import heapq
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        # counts maps each ID to its current frequency in the collection
        counts = {}
        # max_heap stores entries of form (-frequency, id) for lazy retrieval of max
        max_heap = []
        ans = []
        
        for num, delta in zip(nums, freq):
            # update the count for this ID
            new_count = counts.get(num, 0) + delta
            counts[num] = new_count
            
            # push the updated count onto the max-heap (negated for max behavior)
            heapq.heappush(max_heap, (-new_count, num))
            
            # lazily pop entries that are out of date
            while max_heap:
                neg_count, top_id = max_heap[0]
                # current actual count of top_id
                actual_count = counts.get(top_id, 0)
                # if it matches the heap entry, this is our current max
                if -neg_count == actual_count:
                    break
                # otherwise discard it
                heapq.heappop(max_heap)
            
            # determine the current max frequency
            if not max_heap:
                ans.append(0)
            else:
                ans.append(-max_heap[0][0])
        
        return ans