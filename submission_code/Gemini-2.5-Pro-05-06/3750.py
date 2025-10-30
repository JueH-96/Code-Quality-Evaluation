import collections # Use collections.defaultdict
import bisect # Use bisect.bisect_left
from typing import List # For type hints

class Solution:
  def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
    n = len(nums)
    
    # Step 1: Preprocess nums to store indices for each value.
    # value_indices will map a number to a sorted list of its indices in nums.
    value_indices = collections.defaultdict(list)
    for i, num_val in enumerate(nums):
        value_indices[num_val].append(i)
    # The lists in value_indices are already sorted because we iterate i 
    # in increasing order (0, 1, 2, ...) and append to the lists.
    
    ans = [] # This will store the results for each query.
    
    # Step 2: Process each query.
    for q_idx in queries:
        val_at_q_idx = nums[q_idx] # The value at the queried index.
        
        # Get the list of all indices where val_at_q_idx appears.
        # This list is sorted.
        indices_list = value_indices[val_at_q_idx]
        num_occurrences = len(indices_list)
        
        # If the value appears only once (or not at all, though not possible here
        # as q_idx is a valid index and nums is non-empty), there's no *other* index j 
        # with the same value. So, distance is -1.
        if num_occurrences <= 1:
            ans.append(-1)
            continue
        
        # Find the position of q_idx within indices_list using binary search.
        # bisect_left returns an index k_ptr such that all elements in indices_list[:k_ptr]
        # are less than q_idx, and all elements in indices_list[k_ptr:] are greater than or equal to q_idx.
        # Since q_idx is guaranteed to be in indices_list (as nums[q_idx] = val_at_q_idx and
        # indices_list contains all occurrences of val_at_q_idx), indices_list[k_ptr] will be q_idx.
        k_ptr = bisect.bisect_left(indices_list, q_idx)
        
        # Identify the two neighbors of q_idx in the conceptual "circular" list of occurrences.
        # These are the candidates for the closest same-valued element.
        
        # Neighbor "before" q_idx in indices_list (circularly).
        # Example: if indices_list = [idx0, idx1, idx2] and q_idx = idx0 (so k_ptr = 0),
        # the previous element circularly is idx2.
        # The index in indices_list for this previous element is (k_ptr - 1 + num_occurrences) % num_occurrences.
        prev_neighbor_list_idx = (k_ptr - 1 + num_occurrences) % num_occurrences
        prev_neighbor_actual_array_idx = indices_list[prev_neighbor_list_idx]
        
        # Neighbor "after" q_idx in indices_list (circularly).
        # Example: if indices_list = [idx0, idx1, idx2] and q_idx = idx2 (so k_ptr = 2),
        # the next element circularly is idx0.
        # The index in indices_list for this next element is (k_ptr + 1) % num_occurrences.
        next_neighbor_list_idx = (k_ptr + 1) % num_occurrences
        next_neighbor_actual_array_idx = indices_list[next_neighbor_list_idx]
        
        # Calculate circular distances in the nums array from q_idx to these two neighbors.
        # The circular distance between idx1 and idx2 in an array of length n is
        # min(abs(idx1 - idx2), n - abs(idx1 - idx2)).
        
        # Distance to the "previous" neighbor (in the circular list of occurrences).
        dist_to_prev = abs(q_idx - prev_neighbor_actual_array_idx)
        dist_to_prev = min(dist_to_prev, n - dist_to_prev) # Circular distance in nums
        
        # Distance to the "next" neighbor (in the circular list of occurrences).
        dist_to_next = abs(q_idx - next_neighbor_actual_array_idx)
        dist_to_next = min(dist_to_next, n - dist_to_next) # Circular distance in nums
        
        # The answer for the current query is the minimum of these two distances.
        # If num_occurrences is 2, say indices_list = [idxA, idxB].
        # If q_idx = idxA, then k_ptr = 0.
        # prev_neighbor_actual_array_idx will be idxB. 
        # next_neighbor_actual_array_idx will also be idxB.
        # So dist_to_prev and dist_to_next will be the same value: dist(idxA, idxB).
        # min(dist_to_prev, dist_to_next) will correctly yield this distance.
        ans.append(min(dist_to_prev, dist_to_next))
            
    return ans