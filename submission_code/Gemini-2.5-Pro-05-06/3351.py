from typing import List

class Solution:
  def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
    # Sort happiness in descending order.
    # Python's Timsort is O(N log N) time complexity.
    # It can be O(N) space in worst-case for some data patterns, 
    # or O(log N) if the data is structured nicely or for typical cases.
    # If modifying the input list is not allowed, a copy should be made:
    # sorted_happiness = sorted(happiness, reverse=True)
    # For competitive programming, modifying input list is usually fine.
    happiness.sort(reverse=True) 
    
    total_happiness_sum = 0
    
    # We select k children in k turns.
    # In turn i (0-indexed, from 0 to k-1):
    # We pick the child with the (i)-th largest original happiness.
    # This child's original happiness is happiness[i] from the sorted list.
    # By the time we pick this child, i children have already been picked in previous turns.
    # In each of those i turns, this child's happiness was reduced by 1.
    # So, this child's happiness has been reduced by a total of i.
    # The value contributed by this child is happiness[i] - i.
    # If happiness[i] - i becomes non-positive, it contributes 0.
    
    for i in range(k):
      # Original happiness of the child selected in this turn.
      original_h_of_selected_child = happiness[i]
      
      # Happiness lost due to i previous selections.
      decrement_amount = i 
      
      effective_h = original_h_of_selected_child - decrement_amount
      
      if effective_h > 0:
        total_happiness_sum += effective_h
      else:
        # If the current child's effective happiness is 0 or less,
        # any subsequent child we could pick would also yield 0 or less happiness.
        # This is because:
        #   - Their original happiness is less than or equal to the current child's 
        #     (i.e., happiness[j] <= happiness[i] for j > i, due to descending sort).
        #   - The decrement they would suffer is greater (i.e., j > i).
        #   - So, happiness[j] - j < happiness[j] - i <= happiness[i] - i <= 0.
        # Thus, all further children will contribute 0, and we can stop early.
        break
        
    return total_happiness_sum