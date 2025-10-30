from typing import List
from collections import defaultdict
from bisect import bisect_left

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        
        # Step 1: Preprocessing - Store all indices for each value
        # value_to_indices[val] will be a sorted list of indices where 'val' appears
        value_to_indices = defaultdict(list)
        for i in range(n):
            value_to_indices[nums[i]].append(i)
            
        answer = []
        
        # Helper function to calculate circular distance between two indices
        def calculate_circular_dist(idx1: int, idx2: int, array_len: int) -> int:
            """Calculates the minimum circular distance between idx1 and idx2."""
            linear_dist = abs(idx1 - idx2)
            return min(linear_dist, array_len - linear_dist)

        # Step 2: Process each query
        for query_idx in queries:
            current_value = nums[query_idx]
            indices_list = value_to_indices[current_value]
            
            # If the value appears only once, there's no "other" index
            if len(indices_list) <= 1:
                answer.append(-1)
                continue
            
            # Find the position of query_idx in its sorted indices_list
            # bisect_left returns an insertion point. Since query_idx is guaranteed
            # to be in indices_list, indices_list[pos] will be query_idx.
            pos = bisect_left(indices_list, query_idx)
            
            min_dist = float('inf')
            k = len(indices_list) # Number of occurrences of current_value
            
            # Candidate 1: The "next" index in the circular list of occurrences
            # Use modulo k to handle wrap-around from the last element to the first
            next_pos = (pos + 1) % k
            next_idx = indices_list[next_pos]
            min_dist = min(min_dist, calculate_circular_dist(query_idx, next_idx, n))
            
            # Candidate 2: The "previous" index in the circular list of occurrences
            # Use (pos - 1 + k) % k to handle wrap-around from the first element to the last
            prev_pos = (pos - 1 + k) % k
            prev_idx = indices_list[prev_pos]
            min_dist = min(min_dist, calculate_circular_dist(query_idx, prev_idx, n))
            
            answer.append(min_dist)
            
        return answer