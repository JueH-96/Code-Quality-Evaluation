import collections # This import is not used and can be removed if desired.
from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        """
        Calculates the number of adjacent elements with the same color after each query.

        The algorithm maintains the current state of the array `nums` and the
        current count of adjacent same-colored pairs `current_adjacent_count`.
        For each query, it determines how the coloring change at the specified index
        affects the adjacent pairs involving that index. It calculates the net change
        in the count (`delta`) by considering the pairs (index-1, index) and 
        (index, index+1), and updates the total count accordingly.

        Args:
            n: The length of the array nums.
            queries: A list of queries, where each query is [index_i, color_i].
                     `index_i` is the 0-based index to color.
                     `color_i` is the new color (a positive integer).

        Returns:
            A list `answer` where `answer[i]` is the count of adjacent elements
            with the same non-zero color after processing the i-th query.
            The count includes the number of indices j such that 0 <= j < n - 1
            and nums[j] == nums[j + 1] and nums[j] != 0.
        """
        
        # Constraints state 1 <= n. If n = 1, there are no adjacent elements, 
        # so the count is always 0.
        if n == 1:
            # Return a list of zeros with the same length as queries.
            return [0] * len(queries)

        # Initialize the array `nums` of size n with all elements uncolored (0).
        nums = [0] * n  
        # Initialize the list to store the answer for each query.
        answer = []
        # Initialize the count of adjacent elements with the same non-zero color.
        current_adjacent_count = 0

        # Process each query one by one.
        for index, new_color in queries:
            
            # Get the color currently at the index before the update.
            old_color = nums[index]

            # Optimization: If the new color is the same as the old color,
            # the state of the array and the count of adjacent pairs do not change.
            # Append the current count and proceed to the next query.
            if old_color == new_color:
                answer.append(current_adjacent_count)
                continue

            # Initialize the change (delta) in the adjacent pair count for this query.
            delta = 0

            # --- Check the impact on the left adjacent pair (index - 1, index) ---
            # This pair exists and needs consideration only if index > 0.
            if index > 0:
                left_neighbor_color = nums[index - 1]
                
                # Check if the pair (index - 1, index) formed a valid adjacent pair *before* the update.
                # A valid pair requires both elements to have the same non-zero color.
                if old_color != 0 and left_neighbor_color == old_color:
                    # If they formed a pair, this pair's contribution is removed due to the color change at 'index'.
                    delta -= 1
                    
                # Check if the pair (index - 1, index) forms a valid adjacent pair *after* the update.
                if new_color != 0 and left_neighbor_color == new_color:
                    # If the new color at 'index' creates a pair with the left neighbor, add its contribution.
                    delta += 1

            # --- Check the impact on the right adjacent pair (index, index + 1) ---
            # This pair exists and needs consideration only if index < n - 1.
            if index < n - 1:
                right_neighbor_color = nums[index + 1]
                
                # Check if the pair (index, index + 1) formed a valid adjacent pair *before* the update.
                if old_color != 0 and right_neighbor_color == old_color:
                    # If they formed a pair, this pair's contribution is removed.
                    delta -= 1
                    
                # Check if the pair (index, index + 1) forms a valid adjacent pair *after* the update.
                if new_color != 0 and right_neighbor_color == new_color:
                    # If the new color at 'index' creates a pair with the right neighbor, add its contribution.
                    delta += 1

            # Apply the calculated change (delta) to the total count of adjacent pairs.
            current_adjacent_count += delta
            
            # Update the color in the array *after* checking neighbors against the old_color.
            nums[index] = new_color
            
            # Append the updated count after this query to the result list.
            answer.append(current_adjacent_count)

        # Return the list containing the counts after each query.
        return answer