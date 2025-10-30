import collections
from typing import List

class Solution:
  """
  Solves the maximum total damage problem using dynamic programming.
  The problem asks for the maximum total damage achievable by casting spells,
  subject to the constraint that if a spell with damage p is cast,
  no spell with damage p-2, p-1, p+1, or p+2 can be cast.
  Each spell can only be cast once. Multiple spells can have the same damage value.
  """
  def maximumTotalDamage(self, power: List[int]) -> int:
    """
    Calculates the maximum possible total damage a magician can cast.

    Args:
      power: A list of integers representing the damage of available spells.

    Returns:
      The maximum possible total damage achievable.
    """
    
    # Step 1: Count frequencies of each power level using collections.Counter.
    # This efficiently groups spells by their damage value and counts occurrences.
    # For example, if power = [1, 1, 3, 4], counts will be {1: 2, 3: 1, 4: 1}.
    counts = collections.Counter(power)
    
    # Constraints state power.length >= 1, so the input list 'power' is never empty, 
    # and thus 'counts' will also not be empty.
        
    # Step 2: Get unique power levels present in the input and sort them in ascending order.
    # Sorting is crucial for the dynamic programming approach as we need to process 
    # power levels in increasing order to build up the solution.
    # For counts = {1: 2, 3: 1, 4: 1}, unique_powers will be [1, 3, 4].
    unique_powers = sorted(counts.keys())
    
    # n stores the number of unique power levels.
    n = len(unique_powers)
    
    # Step 3: Initialize the DP array.
    # dp[i] will store the maximum total damage achievable considering spells
    # with damage values from the set {unique_powers[0], ..., unique_powers[i]}.
    # The size of the dp array is n, one entry for each unique power level.
    dp = [0] * n
    
    # Step 4: Initialize a pointer 'k'.
    # This pointer 'k' helps efficiently find the index of the largest power level unique_powers[k]
    # such that unique_powers[k] < unique_powers[i] - 2. This condition comes from the constraint:
    # If we cast spell p_i, we cannot cast spells p_i-1 and p_i-2. So the previous spell damage p_k
    # must be strictly less than p_i - 2.
    # 'k' starts at -1, indicating initially that there's no such compatible previous power level found.
    k = -1 
    
    # Step 5: Iterate through the sorted unique power levels to compute DP values.
    for i in range(n):
        # p_i is the current unique power level being considered.
        p_i = unique_powers[i]
        
        # Calculate the total damage obtained if we decide to cast all spells with power p_i.
        # This is the damage value p_i multiplied by the number of spells with this damage, counts[p_i].
        current_damage = p_i * counts[p_i]
        
        # Define the target threshold for previous power levels. Based on the constraint,
        # compatible previous spells must have damage less than p_i - 2.
        target = p_i - 2
        
        # Update the pointer 'k' using a two-pointer technique (k increases monotonically).
        # Move 'k' forward as long as the next power level (unique_powers[k+1])
        # satisfies the compatibility condition (is less than the target threshold)
        # and k+1 is a valid index within the bounds considered so far (less than i).
        # This loop finds the largest index k such that unique_powers[k] < target.
        # Because unique_powers is sorted and target increases with i, k never decreases.
        # This makes the search for k efficient, amortized O(1) per iteration of the outer loop.
        while k + 1 < i and unique_powers[k + 1] < target:
            k += 1
        
        # Calculate the maximum damage contributed by compatible previous spells.
        # If k remains -1, it means no previous power level satisfies the condition unique_powers[k] < p_i - 2.
        # In this case, the contribution from previous spells is 0.
        # Otherwise, the maximum damage from compatible previous spells is given by dp[k].
        previous_max_damage = 0
        if k != -1:
            previous_max_damage = dp[k]
        
        # Calculate the total damage for Option 2: Cast spells with power p_i.
        # This is the sum of the damage from spells with power p_i (current_damage)
        # and the maximum damage achievable from compatible previous spells (previous_max_damage).
        option2_damage = previous_max_damage + current_damage
        
        # Calculate the total damage for Option 1: Do not cast spells with power p_i.
        # In this case, the maximum damage is the same as the maximum damage achievable
        # considering spells up to the immediately preceding unique power level unique_powers[i-1].
        # This value is stored in dp[i-1].
        # If i is 0 (the first unique power level), there's no previous power level, so this option yields 0 damage.
        option1_damage = 0
        if i > 0:
            option1_damage = dp[i-1]
        
        # The DP state dp[i] is determined by taking the maximum of the two options:
        # either include spells with power p_i (Option 2) or exclude them (Option 1).
        dp[i] = max(option1_damage, option2_damage)
            
    # Step 6: The final answer is the maximum damage achievable considering all unique power levels.
    # This value is stored in the last element of the DP array, dp[n-1].
    # Since the constraints guarantee n >= 1, accessing dp[n-1] is always valid.
    return dp[n-1]