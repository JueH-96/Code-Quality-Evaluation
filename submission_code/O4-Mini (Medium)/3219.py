from typing import List
import heapq
import collections

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        # Sort the values to discover components based on value gaps
        sorted_vals = sorted(nums)
        comp_id_by_value = {}
        comp = 0
        prev = None
        # Assign a component ID to each value (duplicates stay in the same comp)
        for x in sorted_vals:
            if prev is None:
                # first value
                comp_id_by_value[x] = comp
            else:
                # if gap > limit, start a new component
                if x - prev > limit:
                    comp += 1
                comp_id_by_value[x] = comp
            prev = x
        
        # Collect all values for each component into a min-heap
        comp_heaps = collections.defaultdict(list)
        for x in nums:
            cid = comp_id_by_value[x]
            comp_heaps[cid].append(x)
        for cid in comp_heaps:
            heapq.heapify(comp_heaps[cid])
        
        # Build the answer by, for each position, taking the smallest available
        # value from the same component as the original value at that position
        ans = []
        for x in nums:
            cid = comp_id_by_value[x]
            smallest = heapq.heappop(comp_heaps[cid])
            ans.append(smallest)
        
        return ans