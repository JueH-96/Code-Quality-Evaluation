from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        """
        Calculates the minimum total cost to cut an m x n cake into 1 x 1 pieces.

        Args:
            m: The height of the cake.
            n: The width of the cake.
            horizontalCut: List of costs for horizontal cuts (size m-1).
            verticalCut: List of costs for vertical cuts (size n-1).

        Returns:
            The minimum total cost.
        """
        # Sort cut costs in descending order to pick the most expensive cuts first.
        # This greedy approach works because applying a higher cost cut earlier means
        # it gets multiplied by a smaller number of existing pieces.
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)

        total_cost = 0
        h_ptr = 0  # Pointer to the current most expensive horizontal cut in the sorted list.
        v_ptr = 0  # Pointer to the current most expensive vertical cut in the sorted list.

        # num_h_pieces: Represents the number of horizontal segments (rows of pieces) currently.
        # Initially 1 (the whole cake is one row).
        # Each horizontal cut increases this count by 1.
        # This count is the multiplier for vertical cuts.
        num_h_pieces = 1 
        
        # num_v_pieces: Represents the number of vertical segments (columns of pieces) currently.
        # Initially 1 (the whole cake is one column).
        # Each vertical cut increases this count by 1.
        # This count is the multiplier for horizontal cuts.
        num_v_pieces = 1

        # We need to make m-1 horizontal cuts and n-1 vertical cuts in total.
        # Continue iterating as long as there are cuts remaining in either dimension.
        while h_ptr < m - 1 or v_ptr < n - 1:
            
            # Get the cost of the current most expensive horizontal cut.
            # If all horizontal cuts are done (h_ptr reaches m-1), use a sentinel value (-1)
            # smaller than any possible valid cost (costs are >= 1) so vertical cuts are prioritized.
            current_h_cost = horizontalCut[h_ptr] if h_ptr < m - 1 else -1
            
            # Get the cost of the current most expensive vertical cut.
            # If all vertical cuts are done (v_ptr reaches n-1), use a sentinel value (-1)
            # smaller than any possible valid cost.
            current_v_cost = verticalCut[v_ptr] if v_ptr < n - 1 else -1

            # Greedy choice: Compare the costs of the next available horizontal and vertical cuts.
            # Perform the cut that has the higher cost.
            # If costs are equal (current_h_cost >= current_v_cost), prioritize horizontal.
            # The tie-breaking rule doesn't affect the minimum total cost.
            if current_h_cost >= current_v_cost:
                # Choose the horizontal cut.
                # The cost incurred is the cost of this specific horizontal cut multiplied by the
                # number of vertical pieces it spans across.
                total_cost += current_h_cost * num_v_pieces
                
                # Move the pointer to the next most expensive horizontal cut.
                h_ptr += 1
                
                # Completing a horizontal cut increases the total number of horizontal pieces (rows) by one.
                # This will be the multiplier for subsequent vertical cuts.
                num_h_pieces += 1 
            else: # current_v_cost > current_h_cost
                # Choose the vertical cut.
                # The cost incurred is the cost of this specific vertical cut multiplied by the
                # number of horizontal pieces it spans across.
                total_cost += current_v_cost * num_h_pieces
                
                # Move the pointer to the next most expensive vertical cut.
                v_ptr += 1
                
                # Completing a vertical cut increases the total number of vertical pieces (columns) by one.
                # This will be the multiplier for subsequent horizontal cuts.
                num_v_pieces += 1

        return total_cost