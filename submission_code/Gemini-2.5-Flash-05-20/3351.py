from typing import List

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        # Sort the happiness array in descending order.
        # This ensures that we prioritize selecting children with higher initial happiness values.
        # The child with the highest initial happiness will be chosen first (decrement 0),
        # the second highest next (decrement 1), and so on.
        happiness.sort(reverse=True)
        
        total_happiness_sum = 0
        
        # Iterate k times to select k children.
        # The loop variable 'i' represents the number of children already selected before the current turn.
        # Thus, the happiness of the child picked in the current turn will be reduced by 'i'.
        for i in range(k):
            # Get the original happiness of the child being considered for this turn.
            # Since the array is sorted descending, happiness[i] is the (i+1)-th highest original happiness.
            current_original_happiness = happiness[i]
            
            # Calculate the effective happiness after applying the decrement.
            # The happiness value cannot become negative.
            effective_happiness = current_original_happiness - i
            
            # If the effective happiness is positive, add it to the total sum.
            if effective_happiness > 0:
                total_happiness_sum += effective_happiness
            else:
                # Optimization:
                # Since the 'happiness' array is sorted in descending order,
                # and 'i' (the decrement amount) is strictly increasing with each turn,
                # if the effective happiness for the current child is 0 or less,
                # it means that any subsequent child (happiness[j] for j > i)
                # will also yield an effective happiness of 0 or less (because happiness[j] <= happiness[i]
                # and it will be decremented by an even larger amount j > i).
                # Therefore, we can stop summing as no further positive contributions are possible.
                break
                
        return total_happiness_sum