from typing import List

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        """
        Calculates the maximum sum of happiness by selecting k children.

        Args:
            happiness: A list of integers representing the happiness values of n children.
            k: A positive integer, the number of children to select.

        Returns:
            The maximum sum of happiness values of the selected children.
        """
        # Sort the happiness values in descending order.
        # The greedy strategy is to always pick the child with the currently
        # highest happiness. Since selecting a child reduces the happiness
        # of *all* other children by 1, picking the initially happiest
        # children first means they undergo the fewest reductions before being picked.
        happiness.sort(reverse=True)

        total_happiness_sum = 0

        # Iterate through the top k happiness values from the sorted list.
        # The child at index i (0-indexed) in the sorted list will be the
        # i-th child we pick (after sorting and starting from the highest).
        # By the time we pick this child, i previous children have been picked,
        # reducing this child's happiness by i.
        for i in range(k):
            # Calculate the happiness of the child at the moment they are picked.
            # This is their initial happiness minus the number of previous turns (i).
            # Happiness cannot be negative.
            current_happiness = happiness[i] - i

            # Only add positive happiness values.
            # Using max(0, current_happiness) handles the non-negative constraint.
            total_happiness_sum += max(0, current_happiness)

        return total_happiness_sum