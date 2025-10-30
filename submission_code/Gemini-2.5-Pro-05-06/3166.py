import collections

class Solution:
  def minGroupsForValidAssignment(self, nums: list[int]) -> int:
    counts_map = collections.Counter(nums)
    # counts is a list of frequencies, e.g., [3, 2] for nums = [3,2,3,2,3].
    counts = list(counts_map.values())

    # nums.length >= 1 (from constraints), so counts list is non-empty.
    
    # s is the size of the smaller groups. s+1 is the size of the larger groups.
    # s must be at least 1.
    # The maximum possible value for s is min_freq (the smallest frequency of any number).
    # If s > min_freq, then for the number with count min_freq, we cannot form
    # any group of size s or s+1, because min_freq < s < s+1.
    min_freq = min(counts)
    
    min_total_groups = float('inf')
    
    # Iterate over possible values for s (the size of the smaller groups)
    for s in range(1, min_freq + 1):
        current_total_groups_for_s = 0
        possible_s = True # Flag to check if this s is feasible for all counts
        
        for c_val in counts:
            # Calculate k_this_c_val = ceil(c_val / (s+1))
            # Integer arithmetic for ceil(A/B) is (A + B - 1) // B for positive integers A, B.
            # Here A = c_val, B = s+1. So, (c_val + (s+1) - 1) // (s+1) simplifies to (c_val + s) // (s+1).
            k_this_c_val = (c_val + s) // (s + 1)
            
            # Check if this k_this_c_val is valid: k_this_c_val * s <= c_val
            # This condition means k_this_c_val <= floor(c_val / s).
            if k_this_c_val * s > c_val:
                # This s is not feasible for this c_val as it violates the condition.
                possible_s = False
                break # Stop checking other counts for this s; try the next s.
            
            current_total_groups_for_s += k_this_c_val
        
        if possible_s:
            # If s was feasible for all counts, update the overall minimum.
            min_total_groups = min(min_total_groups, current_total_groups_for_s)
            
    return min_total_groups