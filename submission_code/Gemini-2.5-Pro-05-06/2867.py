from typing import List

class Solution:
  def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
    MOD = 10**9 + 7
    
    # Stores the index of the most recently encountered '1'.
    # Initialized to -1 to signify that no '1' has been seen yet.
    last_one_idx = -1
    
    # The answer. If there is at least one '1', the number of ways is at least 1.
    # This initial value correctly handles the case of a single '1' in the array,
    # as the loop's multiplication part (in the 'else' branch) won't execute for the first '1'.
    # If there's only one '1' overall, 'count' remains 1.
    count = 1
    
    for i, num_val in enumerate(nums):
      if num_val == 1:
        if last_one_idx == -1:
          # This is the first '1' encountered.
          # No gap to calculate yet, just record its position.
          last_one_idx = i
        else:
          # This is a subsequent '1'. A gap exists between this '1'
          # (at index 'i') and the '1' at 'last_one_idx'.
          
          # The number of zeros in this gap is (i - last_one_idx - 1).
          # Each such zero can be appended to the previous good subarray 
          # or prepended to the current good subarray.
          # This means there are (number of zeros + 1) ways to make the cut/split.
          # num_options_for_split = (i - last_one_idx - 1) + 1 = i - last_one_idx.
          num_options_for_split = i - last_one_idx
          
          count = (count * num_options_for_split) % MOD
          
          # Update the position of the last seen '1' to the current index.
          last_one_idx = i 
          
    # After iterating through the array:
    if last_one_idx == -1:
      # No '1's were found in the array.
      # It's impossible to form any good subarray (which requires one '1').
      return 0
    else:
      # At least one '1' was found. 'count' holds the calculated number of ways.
      return count