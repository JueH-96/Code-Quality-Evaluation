from typing import List

class Solution:
  def maximumLength(self, nums: List[int]) -> int:
    # Candidate 1: Subsequence of all even numbers (e.g., E, E, E, ...)
    # Parity of sums: (even + even) % 2 = 0.
    # Length is the total count of even numbers in nums.
    count_even = 0
    
    # Candidate 2: Subsequence of all odd numbers (e.g., O, O, O, ...)
    # Parity of sums: (odd + odd) % 2 = 0.
    # Length is the total count of odd numbers in nums.
    count_odd = 0
    
    # Candidate 3: Subsequence of alternating parities, starting with Even (E, O, E, O, ...).
    # Parity of sums: (even + odd) % 2 = 1 or (odd + even) % 2 = 1.
    # Length found by greedily picking elements.
    len_alternating_eo = 0
    # current_needed_parity_for_eo: parity we are looking for to extend E,O,E... sequence.
    # Starts by needing an Even number (parity 0).
    current_needed_parity_for_eo = 0 
    
    # Candidate 4: Subsequence of alternating parities, starting with Odd (O, E, O, E, ...).
    # Parity of sums: (odd + even) % 2 = 1 or (even + odd) % 2 = 1.
    # Length found by greedily picking elements.
    len_alternating_oe = 0
    # current_needed_parity_for_oe: parity we are looking for to extend O,E,O... sequence.
    # Starts by needing an Odd number (parity 1).
    current_needed_parity_for_oe = 1
    
    for x in nums:
        parity_of_x = x % 2
        
        # Update counts for same-parity subsequences (target sum parity 0)
        if parity_of_x == 0: # x is Even
            count_even += 1
        else: # x is Odd
            count_odd += 1
        
        # Update lengths for alternating-parity subsequence starting E, O, ... (target sum parity 1)
        if parity_of_x == current_needed_parity_for_eo:
            len_alternating_eo += 1
            # Flip the needed parity for the next element in this subsequence type
            current_needed_parity_for_eo = 1 - current_needed_parity_for_eo
        
        # Update lengths for alternating-parity subsequence starting O, E, ... (target sum parity 1)
        if parity_of_x == current_needed_parity_for_oe:
            len_alternating_oe += 1
            # Flip the needed parity for the next element in this subsequence type
            current_needed_parity_for_oe = 1 - current_needed_parity_for_oe
            
    # The final answer is the maximum length found among these four types.
    return max(count_even, count_odd, len_alternating_eo, len_alternating_oe)