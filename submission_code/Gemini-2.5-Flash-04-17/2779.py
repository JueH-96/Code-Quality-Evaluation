from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        """
        Calculates the number of adjacent same-colored non-zero elements
        after each coloring query.

        Args:
            n: The length of the array nums.
            queries: A list of queries, where each query is [index_i, color_i].

        Returns:
            A list of integers, where the i-th element is the number of
            adjacent same-colored non-zero elements after the i-th query.
        """
        # Initialize the array nums with 0s.
        # 0 represents an uncolored element.
        nums = [0] * n
        
        # Initialize the list to store the answer for each query.
        answer = []
        
        # Initialize the count of adjacent elements with the same non-zero color.
        current_count = 0

        # Process each query in the queries list.
        for query in queries:
            idx, new_c = query
            
            # Get the old color at the current index before updating.
            old_c = nums[idx]

            # When the color at index `idx` changes from `old_c` to `new_c`,
            # it can affect the adjacent pairs (idx - 1, idx) and (idx, idx + 1).
            # We first check if the old color at `idx` was part of any
            # same-colored non-zero adjacent pairs involving `idx` and subtract
            # their contribution from the count.

            # Check the left adjacent pair (idx - 1, idx) if idx > 0.
            # If nums[idx - 1] and old_c were the same non-zero color,
            # this adjacency is potentially broken by the color change.
            # Check if nums[idx - 1] == old_c and old_c is not 0.
            if idx > 0:
                if nums[idx - 1] == old_c and old_c != 0:
                    current_count -= 1
                    
            # Check the right adjacent pair (idx, idx + 1) if idx < n - 1.
            # If old_c and nums[idx + 1] were the same non-zero color,
            # this adjacency is potentially broken by the color change.
            # Check if old_c == nums[idx + 1] and old_c is not 0.
            if idx < n - 1:
                if old_c == nums[idx + 1] and old_c != 0:
                    current_count -= 1

            # Update the color at the current index with the new color.
            nums[idx] = new_c

            # After updating, check the adjacent pairs again with the new color.
            # If these pairs now form a same-colored non-zero adjacency,
            # increment the count.

            # Check the left adjacent pair (idx - 1, idx) if idx > 0.
            # If nums[idx - 1] and new_c are the same color, and new_c is not 0,
            # a new valid adjacency is formed or maintained.
            # Constraint 1 <= color_i <= 10^5 ensures new_c is never 0,
            # so we only need to check if nums[idx - 1] == new_c.
            if idx > 0:
                if nums[idx - 1] == new_c:
                    current_count += 1
                    
            # Check the right adjacent pair (idx, idx + 1) if idx < n - 1.
            # If new_c and nums[idx + 1] are the same color, and new_c is not 0,
            # a new valid adjacency is formed or maintained.
            # Again, new_c is never 0.
            if idx < n - 1:
                if new_c == nums[idx + 1]:
                    current_count += 1

            # The current_count now reflects the number of adjacent same-colored
            # non-zero elements after this query. Add it to the answer list.
            answer.append(current_count)

        # Return the list of counts after each query.
        return answer