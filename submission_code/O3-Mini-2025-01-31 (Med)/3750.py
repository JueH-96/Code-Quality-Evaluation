from typing import List
import bisect

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        # Build a dictionary that maps each value to the list of its indices in sorted order.
        indices_map = {}
        for i, val in enumerate(nums):
            if val in indices_map:
                indices_map[val].append(i)
            else:
                indices_map[val] = [i]
        
        result = []
        for q in queries:
            val = nums[q]
            pos_list = indices_map[val]
            # If there's only one occurrence, answer is -1.
            if len(pos_list) < 2:
                result.append(-1)
                continue
            
            # Since pos_list is sorted, find the index of the query position.
            idx = bisect.bisect_left(pos_list, q)
            # Get the left neighbor in circular list.
            left_index = pos_list[idx - 1] if idx > 0 else pos_list[-1]
            # Get the right neighbor in circular list.
            right_index = pos_list[idx + 1] if idx < len(pos_list) - 1 else pos_list[0]
            
            # Calculate circular distance helper function.
            def circ_dist(a, b):
                diff = abs(a - b)
                return min(diff, n - diff)
            
            # Minimum distance from the current query index to its two neighbors.
            min_distance = min(circ_dist(q, left_index), circ_dist(q, right_index))
            result.append(min_distance)
        
        return result