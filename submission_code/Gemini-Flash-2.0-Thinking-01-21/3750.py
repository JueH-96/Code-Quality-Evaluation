from typing import List
from collections import defaultdict
from bisect import bisect_left

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        
        # Preprocess: Store indices for each value
        # Using defaultdict automatically creates an empty list for a new key
        # The lists of indices will be sorted because we iterate through nums from 0 to n-1
        value_to_indices = defaultdict(list)
        for i, num in enumerate(nums):
            value_to_indices[num].append(i)
            
        answer = []
        
        # Process each query
        for query_index in queries:
            val = nums[query_index]
            
            # Get the list of indices where 'val' appears.
            indices_list = value_to_indices[val]
            
            # If the value appears only once, there is no *other* index with the same value
            if len(indices_list) <= 1:
                answer.append(-1)
                continue
                
            # Find the position of query_index within the sorted indices_list.
            # bisect_left finds the insertion point to maintain sorted order.
            # Since query_index is guaranteed to be in indices_list, k will be its index.
            k = bisect_left(indices_list, query_index)
            
            # Get the indices immediately before and after the current index (k)
            # in the circular list of indices where 'val' appears.
            # Use modulo arithmetic for circularity within indices_list.
            prev_k_index = (k - 1 + len(indices_list)) % len(indices_list)
            next_k_index = (k + 1) % len(indices_list)
            
            prev_idx = indices_list[prev_k_index]
            next_idx = indices_list[next_k_index]
            
            # Calculate circular distances from query_index to prev_idx and next_idx
            # Distance between two indices i and j in a circular array of size n
            # is min(abs(i-j), n - abs(i-j))
            
            dist1 = min(abs(query_index - prev_idx), n - abs(query_index - prev_idx))
            dist2 = min(abs(query_index - next_idx), n - abs(query_index - next_idx))
            
            # The minimum distance to any other index is the minimum of these two adjacent indices
            answer.append(min(dist1, dist2))
            
        return answer