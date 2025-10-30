from typing import List

class Solution:
  def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
    n = len(usageLimits)
    
    # Sort usageLimits in non-decreasing order.
    # This allows us to process items with smaller limits first.
    usageLimits.sort()
    
    num_groups_formed = 0
    # current_available_slots stores the sum of capacities of items processed so far,
    # minus the sum of lengths of groups already formed.
    # Effectively, it's the total "slots" or "participations" that the items
    # s_0, ..., s_i (processed so far) can offer, beyond what's needed for
    # the 'num_groups_formed' groups.
    current_available_slots = 0
    
    # Iterate through the sorted usage limits.
    # usageLimits[i] is the capacity of the i-th item (after sorting by capacity).
    # After processing usageLimits[i], we have (i+1) distinct items available
    # (those corresponding to usageLimits[0]...usageLimits[i]).
    for i in range(n):
      # Add the capacity (number of times this item can be used) of the current item.
      current_available_slots += usageLimits[i]
      
      # We are trying to form the (num_groups_formed + 1)-th group.
      # The length of this group must be at least (num_groups_formed + 1)
      # because group lengths must strictly increase. We aim for the minimal
      # possible length for this new group, which is (num_groups_formed + 1).
      target_length_for_next_group = num_groups_formed + 1
      
      # To form this group, two conditions must be met:
      # 1. Distinct items: We need target_length_for_next_group distinct items.
      #    We have processed items usageLimits[0] through usageLimits[i], so we have (i+1) distinct items.
      num_distinct_items_processed = i + 1
      
      # 2. Capacity: We need target_length_for_next_group total "slots" or "participations".
      #    current_available_slots represents the total number of participations available from
      #    items usageLimits[0]...usageLimits[i], after accounting for participations
      #    used by previously formed groups.
      
      if num_distinct_items_processed >= target_length_for_next_group and \
         current_available_slots >= target_length_for_next_group:
        # Both conditions are met, so we can form this new group.
        current_available_slots -= target_length_for_next_group # Allocate slots for this group
        num_groups_formed += 1
        
    return num_groups_formed