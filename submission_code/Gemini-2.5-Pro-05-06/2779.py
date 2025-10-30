from typing import List

class Solution:
  def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
    nums = [0] * n  # Represents the colors, 0 for uncolored
    current_adjacent_pairs = 0
    answer: List[int] = [] # To store results for each query

    for idx, new_color in queries:
        # Get the color currently at nums[idx] before changing it.
        # This is important to determine if an existing pair is broken.
        old_color = nums[idx]

        # If the color at idx doesn't actually change, then the number of
        # adjacent pairs also doesn't change.
        # Note: problem constraints state 1 <= color_i <= 10^5, so new_color is never 0.
        if new_color == old_color:
            answer.append(current_adjacent_pairs)
            continue

        # Check impact on the left potential pair: (nums[idx-1], nums[idx])
        # This check is only needed if idx is not the first element.
        if idx > 0:
            left_neighbor_color = nums[idx-1]
            
            # If (left_neighbor_color, old_color) was a counted pair, decrement.
            # A pair (V, W) is counted if V == W and V != 0 (and W != 0).
            # If left_neighbor_color == old_color and old_color != 0, then
            # (left_neighbor_color, old_color) was a pair like (X,X) with X!=0.
            # This pair is now broken or changed due to nums[idx] changing.
            if left_neighbor_color == old_color and old_color != 0:
                current_adjacent_pairs -= 1
            
            # If (left_neighbor_color, new_color) is a counted pair, increment.
            # new_color is guaranteed to be != 0 by problem constraints.
            # If left_neighbor_color == new_color, then
            # (left_neighbor_color, new_color) forms a pair (Y,Y) with Y!=0.
            # This pair is newly formed or re-established.
            if left_neighbor_color == new_color: # new_color != 0 is guaranteed
                current_adjacent_pairs += 1
        
        # Check impact on the right potential pair: (nums[idx], nums[idx+1])
        # This check is only needed if idx is not the last element.
        if idx < n - 1:
            right_neighbor_color = nums[idx+1]

            # If (old_color, right_neighbor_color) was a counted pair, decrement.
            # Similar logic as for the left pair.
            if old_color == right_neighbor_color and old_color != 0:
                current_adjacent_pairs -= 1
            
            # If (new_color, right_neighbor_color) is a counted pair, increment.
            # Similar logic as for the left pair.
            if new_color == right_neighbor_color: # new_color != 0 is guaranteed
                current_adjacent_pairs += 1
        
        # Actually update the color in the array for subsequent queries
        nums[idx] = new_color
        
        # Record the count of adjacent pairs after this query
        answer.append(current_adjacent_pairs)
            
    return answer