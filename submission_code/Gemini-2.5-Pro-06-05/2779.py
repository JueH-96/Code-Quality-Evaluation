from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        """
        Calculates the number of adjacent elements with the same color after each query.

        Args:
            n: The length of the array.
            queries: A list of queries, where each query is [index, color].

        Returns:
            A list where the i-th element is the count of adjacent same-colored
            pairs after the i-th query.
        """
        # nums array stores the colors of each index. Initially all 0 (uncolored).
        nums = [0] * n
        # answer list stores the result after each query.
        answer = []
        # current_pairs_count stores the number of adjacent elements with the same, non-zero color.
        current_pairs_count = 0

        for index, color in queries:
            # Get the color that was at the index before this query.
            old_color = nums[index]
            
            # If the new color is the same as the old one, the count of pairs does not change.
            # Since problem constraints state color_i >= 1, old_color == color implies the
            # cell was already colored with a non-zero color.
            if old_color == color:
                answer.append(current_pairs_count)
                continue

            # Update count based on the left neighbor, if it exists.
            if index > 0:
                left_neighbor_color = nums[index - 1]
                # A non-zero neighbor is required to form a counted pair.
                if left_neighbor_color != 0:
                    # If a pair existed with the old color, it's now broken.
                    if left_neighbor_color == old_color:
                        current_pairs_count -= 1
                    # If a new pair is formed with the new color, count it.
                    if left_neighbor_color == color:
                        current_pairs_count += 1
            
            # Update count based on the right neighbor, if it exists.
            if index < n - 1:
                right_neighbor_color = nums[index + 1]
                # A non-zero neighbor is required to form a counted pair.
                if right_neighbor_color != 0:
                    # If a pair existed with the old color, it's now broken.
                    if right_neighbor_color == old_color:
                        current_pairs_count -= 1
                    # If a new pair is formed with the new color, count it.
                    if right_neighbor_color == color:
                        current_pairs_count += 1

            # Apply the color change to the array.
            nums[index] = color
            
            # Record the new count of adjacent pairs.
            answer.append(current_pairs_count)

        return answer