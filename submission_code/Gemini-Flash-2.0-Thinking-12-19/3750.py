from typing import List

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        """
        Finds the minimum circular distance to another element with the same value
        for each query index in a circular array.

        Args:
            nums: A list of integers representing the circular array.
            queries: A list of integers representing the query indices.

        Returns:
            A list of integers where the i-th element is the minimum circular
            distance for query i, or -1 if no other element with the same value exists.
        """
        n = len(nums)
        
        # 1. Build dictionary value_indices mapping each value to a list of indices
        #    where it appears.
        #    value_indices: key = value, value = list of indices where value appears
        value_indices = {}
        for i in range(n):
            val = nums[i]
            if val not in value_indices:
                value_indices[val] = []
            value_indices[val].append(i)
            
        # 2. Sort the list of indices for each value and build the
        #    original_index_to_list_position map.
        #    This map stores the position of each original index within the
        #    sorted list of indices for its corresponding value.
        #    original_index_to_list_position: key = original index, value = its position in the *sorted* list for its value
        original_index_to_list_position = {}
        
        for val, indices_list in value_indices.items():
            # Sort the list of indices where this value appears
            indices_list.sort() 
            
            # Populate the position map for each original index
            # This allows O(1) lookup of an index's position within its value's
            # sorted list of indices during query processing.
            for p, index in enumerate(indices_list):
                original_index_to_list_position[index] = p
        
        # Helper function to calculate circular distance between two indices
        def circular_distance(idx1: int, idx2: int, array_len: int) -> int:
            """Calculates the minimum circular distance between two indices."""
            # The distance in one direction (linear)
            linear_dist = abs(idx1 - idx2)
            # The distance in the other direction (wrapping around)
            wrap_dist = array_len - linear_dist
            return min(linear_dist, wrap_dist)
            
        # 3. Process queries
        answer = []
        for q_idx in queries:
            target_val = nums[q_idx]
            
            # Get the sorted list of indices for the target value
            # Since q_idx is a valid index (0 <= q_idx < n), nums[q_idx] is a value
            # present in the array, so it must be a key in value_indices.
            indices_list = value_indices[target_val]
            k = len(indices_list) # Number of occurrences of target_val
            
            if k <= 1:
                # If the value appears only once, there is no other index with the same value
                answer.append(-1)
            else:
                # Get the position of q_idx within the sorted indices_list
                # This position was precomputed and stored in the map
                p = original_index_to_list_position[q_idx] # O(1) lookup
                
                # Find the previous and next indices in the circular sorted list of indices.
                # These are the two candidates for the nearest occurrence in the circular array.
                # Use modulo k to handle wrapping around the list of indices itself.
                prev_idx = indices_list[(p - 1 + k) % k]
                next_idx = indices_list[(p + 1) % k]
                
                # Calculate the circular distances from the query index (q_idx)
                # to these two candidate indices (prev_idx and next_idx)
                # in the original circular array nums of size n.
                dist_prev = circular_distance(q_idx, prev_idx, n)
                dist_next = circular_distance(q_idx, next_idx, n)
                
                # The minimum of these two distances is the minimum circular distance
                # to any other index with the same value.
                answer.append(min(dist_prev, dist_next))
                
        return answer