from typing import List

class Solution:
  def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
    """
    Calculates the maximum sum of happiness values of k selected children
    using a greedy approach.
    """
    
    # To maximize the sum, we should always pick the child who is currently
    # the happiest. This corresponds to picking children in descending order
    # of their initial happiness.
    
    # Sort happiness in descending order.
    happiness.sort(reverse=True)
    
    total_happiness = 0
    
    # We select k children over k turns.
    for i in range(k):
        # The happiness of the child picked at turn `i` (0-indexed) is
        # their original happiness minus the number of turns already passed (`i`).
        current_happiness = happiness[i] - i
        
        # The happiness value cannot become negative. If it is non-positive,
        # we can stop, as all subsequent children will also contribute 0 happiness
        # due to the sorted order.
        if current_happiness > 0:
            total_happiness += current_happiness
        else:
            # Since the array is sorted, if we get a non-positive
            # happiness value, all subsequent values will also be non-positive.
            break
            
    return total_happiness