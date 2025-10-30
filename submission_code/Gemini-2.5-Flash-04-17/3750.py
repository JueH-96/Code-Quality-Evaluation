from typing import List
from collections import defaultdict
from bisect import bisect_left

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        
        # Pre-process: map values to sorted list of indices
        # Using defaultdict simplifies adding indices to lists
        value_indices = defaultdict(list)
        for i, num in enumerate(nums):
            value_indices[num].append(i)
            
        answer = []
        for q_i in queries:
            # Get the value at the query index
            val = nums[q_i]
            
            # Get the list of all indices where this value appears
            # Use .get() with a default empty list just in case, though
            # queries[i] is a valid index, so nums[queries[i]] exists,
            # and will be in value_indices unless nums was empty (constraint 1 <= nums.length)
            indices_of_val = value_indices.get(val, [])
            
            # If the value appears only once, there is no *other* index
            if len(indices_of_val) <= 1:
                answer.append(-1)
                continue
                
            # Find the position of q_i within the sorted list indices_of_val
            # bisect_left finds the insertion point for q_i. Since q_i is
            # guaranteed to be in indices_of_val, this returns the index
            # of q_i in that sorted list.
            # Note: bisect_left(a, x) returns the index of the first element >= x.
            # If x is in a, this is its index.
            p = bisect_left(indices_of_val, q_i)
            
            # The length of the list of indices for the current value
            k = len(indices_of_val)
            
            # Find the indices in the original array 'nums' that are the
            # immediate neighbors of q_i in the *circular* list of indices_of_val.
            
            # The index in indices_of_val before p (circularly)
            # (p - 1 + k) % k handles wrap-around correctly for p=0.
            p_prev = (p - 1 + k) % k
            # The actual index in nums
            idx_prev = indices_of_val[p_prev]
            
            # The index in indices_of_val after p (circularly)
            # (p + 1) % k handles wrap-around correctly for p=k-1.
            p_next = (p + 1) % k
            # The actual index in nums
            idx_next = indices_of_val[p_next]
            
            # Calculate the circular distance between q_i and idx_prev
            # Linear distance is abs(q_i - idx_prev)
            # Circular distance is min(linear distance, n - linear distance)
            dist_prev_linear = abs(q_i - idx_prev)
            circular_dist_prev = min(dist_prev_linear, n - dist_prev_linear)
            
            # Calculate the circular distance between q_i and idx_next
            dist_next_linear = abs(q_i - idx_next)
            circular_dist_next = min(dist_next_linear, n - dist_next_linear)
            
            # The minimum distance to any other index with the same value
            # is the minimum of the distances to these two neighbors.
            answer.append(min(circular_dist_prev, circular_dist_next))
            
        return answer