class Solution:
  def sumImbalanceNumbers(self, nums: list[int]) -> int:
    n = len(nums)
    total_imbalance_sum = 0

    # Iterate over all possible subarrays nums[i..j]
    # Outer loop for the start of the subarray
    for i in range(n):
      # For each starting point i, we extend the subarray to the right with j.
      # 'seen' tracks elements in the current subarray nums[i..j].
      # Values in nums are from 1 to n.
      # We make 'seen' array of size n+2 for 0-indexed access (e.g., seen[value]).
      # seen[0] and seen[n+1] act as sentinels (always False as 0 and n+1 are outside [1,n]).
      seen = [False] * (n + 2) 
      
      num_distinct = 0
      # count_adjacent_pairs stores the number of elements x such that
      # both x and x+1 are present in the current set of unique elements from nums[i..j].
      count_adjacent_pairs = 0 
      
      # Inner loop for the end of the subarray
      for j in range(i, n):
        val = nums[j]
        
        if not seen[val]:
          # val is a new distinct element for the current subarray nums[i..j]
          seen[val] = True
          num_distinct += 1
          
          # Update count_adjacent_pairs based on the new element val.
          # These checks effectively query OldSet membership for val-1 and val+1,
          # because seen[val] was just set, but seen[val-1] and seen[val+1] reflect state before val.
          
          # If val-1 is present, (val-1, val) becomes an adjacent pair.
          if seen[val-1]: # True if val-1 was in OldSet (and val-1 > 0).
            count_adjacent_pairs += 1
          
          # If val+1 is present, (val, val+1) becomes an adjacent pair.
          if seen[val+1]: # True if val+1 was in OldSet (and val+1 <= N).
            count_adjacent_pairs += 1
        
        # Whether val was new or already seen, calculate imbalance for current set nums[i..j]
        # and add to total_imbalance_sum.
        # The imbalance number is (num_distinct - 1) - count_adjacent_pairs.
        # This is 0 if num_distinct <= 1.
        if num_distinct <= 1:
          # Imbalance is 0 for subarrays with 0 or 1 distinct element.
          pass 
        else:
          # For num_distinct >= 2
          current_imbalance = (num_distinct - 1) - count_adjacent_pairs
          total_imbalance_sum += current_imbalance
          
    return total_imbalance_sum