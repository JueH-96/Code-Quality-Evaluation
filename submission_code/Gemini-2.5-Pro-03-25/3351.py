import math # Importing math is not strictly necessary for this solution, but generally good practice.
from typing import List # Required for type hints like List[int].

class Solution:
    """
    This class provides a solution to find the maximum happiness sum
    achievable by selecting k children according to the specified rules.
    """
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        """
        Calculates the maximum sum of happiness values of k selected children.

        The selection process involves k turns. In each turn, one child is selected,
        and the happiness of all remaining unselected children decreases by 1 (cannot go below 0).
        To maximize the total happiness sum, we should always pick the child with the currently
        highest happiness value. This greedy strategy works because picking a higher happiness
        value now contributes more to the sum immediately. While this decreases other children's
        happiness, the decrease is uniform (-1), so the relative order of happiness among the
        remaining children is maintained (or values become 0).

        Therefore, the optimal strategy is to consider the children in descending order of their
        initial happiness. We select the top k children based on their initial happiness.
        The happiness contributed by the i-th selected child (0-indexed) will be their initial
        happiness minus the number of turns already passed (which is i), clamped at 0.

        Args:
            happiness: A list of integers representing the initial happiness values of n children.
            k: The number of children to select.

        Returns:
            The maximum achievable sum of happiness values for the selected children.
        """
        
        # Sort the happiness list in descending order. This allows us to efficiently
        # identify and process the children with the highest initial happiness values first.
        happiness.sort(reverse=True)
        
        total_happiness_sum = 0
        
        # Iterate k times to simulate the selection process for k turns.
        # The loop index 'i' also represents the number of turns that have passed (0-indexed)
        # before the current selection. This value 'i' is the amount by which the happiness
        # of the child being considered in this iteration has decreased.
        for i in range(k):
            # Get the initial happiness of the child considered in this turn (the i-th highest).
            initial_happiness = happiness[i]
            
            # Calculate the happiness of this child at the moment of selection.
            # It's their initial happiness minus the number of previous selections (i).
            happiness_at_selection = initial_happiness - i
            
            # Check if the calculated happiness is positive. Happiness cannot be negative.
            if happiness_at_selection > 0:
                # If positive, add this value to the total sum.
                total_happiness_sum += happiness_at_selection
            else:
                # If the happiness value at selection is zero or negative,
                # any subsequent child selected will also contribute zero or negative happiness
                # when adjusted for the turns passed. This is because:
                # 1. The initial happiness values are non-increasing (due to sorting: happiness[j] <= happiness[i] for j > i).
                # 2. The decrement factor 'j' is increasing (j > i).
                # Therefore, happiness[j] - j <= happiness[i] - j < happiness[i] - i <= 0 for j > i.
                # We can stop the selection process early as further selections will add 0 to the sum.
                break
                
        # Return the total maximum happiness sum accumulated.
        return total_happiness_sum