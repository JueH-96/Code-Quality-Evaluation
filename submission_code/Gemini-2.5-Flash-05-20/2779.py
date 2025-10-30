from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize the array 'nums' with all elements uncolored (0).
        # A value of 0 indicates an uncolored element.
        nums = [0] * n
        
        # Initialize the count of adjacent elements with the same non-zero color.
        current_same_color_pairs_count = 0
        
        # This list will store the answer for each query.
        answer = []

        # Process each query
        for query in queries:
            index, color = query[0], query[1]
            
            # Store the old color at the current index before updating.
            old_color = nums[index]

            # Step 1: Before updating, check if the original color at 'index'
            # formed any valid same-colored pairs with its neighbors.
            # If so, these pairs will be broken, so decrement the count.
            
            # Check the left neighbor: (nums[index-1], old_color)
            # Ensure index is not 0 (i.e., there is a left neighbor).
            if index > 0:
                # A pair counts if both elements are the same AND they are not uncolored (0).
                # Here, we check if nums[index-1] was the same as old_color AND old_color was non-zero.
                if nums[index - 1] == old_color and old_color != 0:
                    current_same_color_pairs_count -= 1
            
            # Check the right neighbor: (old_color, nums[index+1])
            # Ensure index is not n-1 (i.e., there is a right neighbor).
            if index < n - 1:
                # A pair counts if both elements are the same AND they are not uncolored (0).
                # Here, we check if old_color was the same as nums[index+1] AND old_color was non-zero.
                if old_color == nums[index + 1] and old_color != 0:
                    current_same_color_pairs_count -= 1
            
            # Step 2: Update the color at the current index.
            nums[index] = color

            # Step 3: After updating, check if the new color at 'index'
            # forms any new valid same-colored pairs with its neighbors.
            # If so, increment the count.
            # Note: 'color' is guaranteed by constraints (1 <= color_i <= 10^5) to be >= 1,
            # so we don't need to explicitly check 'color != 0' in these conditions.
            
            # Check the left neighbor: (nums[index-1], new_color)
            # Ensure index is not 0.
            if index > 0:
                # If nums[index-1] is the same as the new 'color', a new valid pair is formed.
                if nums[index - 1] == color:
                    current_same_color_pairs_count += 1
            
            # Check the right neighbor: (new_color, nums[index+1])
            # Ensure index is not n-1.
            if index < n - 1:
                # If the new 'color' is the same as nums[index+1], a new valid pair is formed.
                if color == nums[index + 1]:
                    current_same_color_pairs_count += 1
            
            # Append the current total count of valid adjacent pairs to the answer list.
            answer.append(current_same_color_pairs_count)
        
        # Return the final list of counts after all queries have been processed.
        return answer