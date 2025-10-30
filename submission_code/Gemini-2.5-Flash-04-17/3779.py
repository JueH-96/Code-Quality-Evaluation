from typing import List

class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        """
        Calculates the maximum total weight gained by eating pizzas optimally.

        Args:
            pizzas: A list of integers representing the weights of the pizzas.
                    n = len(pizzas) is a multiple of 4.

        Returns:
            The maximum total weight gained.
        """
        # Sort the pizzas by weight in ascending order.
        # Sorting the pizzas helps in identifying the smallest and largest weights easily,
        # which is crucial for determining the Z and Y values in each group.
        # This step takes O(n log n) time, where n is the number of pizzas.
        pizzas.sort()
        
        n = len(pizzas)
        total_gain = 0
        
        # There are n/4 days in total. On each day, exactly 4 pizzas are eaten.
        # The weight gained depends on whether the day is odd or even:
        # - On odd-numbered days, the weight gained is Z (the largest weight in the group).
        # - On even-numbered days, the weight gained is Y (the second largest weight in the group).
        # The goal is to maximize the sum of these n/4 contributing weights.
        #
        # By sorting the pizzas `p_0, p_1, ..., p_{n-1}` (where p_0 <= p_1 <= ... <= p_{n-1}),
        # we can strategically form groups of 4 and assign them to days to maximize the sum.
        #
        # Based on the problem structure and example outcomes, an optimal strategy
        # appears to select specific pizzas from the sorted list to be the ones
        # whose weights contribute to the total gain. These contributing pizzas are
        # the 1st, 3rd, 5th, and so on, up to the (n/2 - 1)-th largest pizzas.
        # There are exactly n/4 such pizzas, matching the number of days (and contributions).
        #
        # In a 0-indexed sorted list `p_0, p_1, ..., p_{n-1}`, the k-th largest pizza
        # (where k is 1-indexed) is found at index `n - k`.
        #
        # The indices of the pizzas contributing to the maximum total gain are:
        # - For the 1st largest pizza (k=1): index `n - 1`. This pizza will likely be the Z on Day 1 (an odd day).
        # - For the 3rd largest pizza (k=3): index `n - 3`. This pizza will likely be the Y on Day 2 (an even day).
        # - For the 5th largest pizza (k=5): index `n - 5`. This pizza will likely be the Z on Day 3 (an odd day).
        # - And so on.
        #
        # The index for the (2*i + 1)-th largest pizza (where i is 0-indexed, representing the i-th pair in the pattern 1st, 3rd, 5th...) is `n - (2*i + 1)`, which simplifies to `n - 1 - 2*i`.
        # We need to sum the weights of these pizzas for i = 0, 1, ..., n/4 - 1.
        # The number of iterations (n/4) corresponds exactly to the number of days and thus the number of pizza weights added to the total gain.
        
        # Iterate i from 0 up to n/4 - 1. Each value of i corresponds to one contribution
        # to the total gain following the identified pattern.
        for i in range(n // 4):
            # Calculate the 0-indexed position in the sorted list of the pizza
            # that contributes to the gain for the current step `i`.
            # The indices are `n-1`, `n-3`, `n-5`, etc. as i goes from 0 upwards.
            gain_index = n - 1 - 2 * i
            
            # Add the weight of the pizza at this calculated index to the total gain.
            total_gain += pizzas[gain_index]
        
        # The loop performs n/4 additions and index calculations, taking O(n) time.
        # The overall time complexity is dominated by the initial sorting step: O(n log n).
        # The space complexity is O(n) or O(log n) depending on the space used by the sorting algorithm.
        
        return total_gain