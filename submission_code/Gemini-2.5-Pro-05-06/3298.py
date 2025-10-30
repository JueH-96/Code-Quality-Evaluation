import collections
from typing import List

class Solution:
  def maxSelectedElements(self, nums: List[int]) -> int:
    # Sort the numbers to process them in increasing order.
    # This allows the dynamic programming approach where we build upon
    # previously computed sequence lengths.
    nums.sort()
    
    # dp[val] stores the maximum length of a consecutive sequence
    # ending with the number 'val'.
    # We use defaultdict(int) so that if a key is accessed and not present,
    # it defaults to 0. This is convenient for dp[val-1] or dp[val]
    # when val-1 or val are not yet in dp (meaning no sequence ends there yet,
    # so a new sequence starting with val-1 or val would have length 0+1=1).
    dp = collections.defaultdict(int)
    
    # Initialize max_len. Since constraints state nums.length >= 1,
    # we can always select at least one element, so the minimum possible
    # max_len is 1.
    max_overall_length = 1
            
    for num_val in nums:
      # For the current number `num_val` from the sorted list, we have two choices
      # for the value it will contribute to a sequence:
      # 1. Use `num_val` as its original value: `num_val`.
      # 2. Use `num_val` by incrementing it to `num_val + 1`.

      # Calculate the length of the sequence if we use `num_val` to form `num_val + 1`.
      # This sequence would extend a sequence ending at `num_val`.
      # The length would be (length of sequence ending at `num_val`) + 1.
      # `dp[num_val]` here refers to sequences ending at `num_val` formed by *previous* elements in `nums`.
      len_val_incremented = dp[num_val] + 1
      
      # Calculate the length of the sequence if we use `num_val` to form `num_val` itself.
      # This sequence would extend a sequence ending at `num_val - 1`.
      # The length would be (length of sequence ending at `num_val - 1`) + 1.
      # `dp[num_val - 1]` here refers to sequences ending at `num_val - 1` formed by *previous* elements.
      len_val_as_is = dp[num_val - 1] + 1
      
      # Update the dp table.
      # It's crucial that `len_val_incremented` and `len_val_as_is` are computed
      # *before* `dp[num_val]` or `dp[num_val+1]` are updated for the current `num_val`.
      # This ensures that `dp[num_val]` (used for `len_val_incremented`) and
      # `dp[num_val-1]` (used for `len_val_as_is`) reflect states from processing
      # elements *prior* to the current `num_val`.

      # If we choose to use the current `num_val` to form the value `num_val + 1`:
      # The sequence ending at `num_val + 1` will have length `len_val_incremented`.
      # We update `dp[num_val + 1]` by taking the maximum of its current value (if any, else 0) 
      # and this new potential length.
      dp[num_val + 1] = max(dp[num_val + 1], len_val_incremented)
      
      # If we choose to use the current `num_val` to form the value `num_val` itself:
      # The sequence ending at `num_val` will have length `len_val_as_is`.
      # We update `dp[num_val]` similarly.
      dp[num_val] = max(dp[num_val], len_val_as_is)
      
      # Update the overall maximum length found so far.
      # The new maximum could be from a sequence ending at `num_val` (value `num_val` itself)
      # or `num_val + 1` (value `num_val` incremented).
      max_overall_length = max(max_overall_length, dp[num_val], dp[num_val + 1])
      
    return max_overall_length