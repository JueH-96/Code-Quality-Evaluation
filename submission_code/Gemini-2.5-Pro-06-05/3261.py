from typing import List

class Solution:
  def minOrAfterOperations(self, nums: List[int], k: int) -> int:
    """
    Finds the minimum possible bitwise OR of the remaining elements of nums
    after applying at most k operations.
    """

    def check(target_or_val: int) -> bool:
      """
      Checks if it's possible to achieve a final OR value that is a submask of
      target_or_val using at most k operations.
      """
      # We count the number of operations needed to ensure every final number `x`
      # satisfies (x | target_or_val) == target_or_val.
      ops_needed = 0
      # current_and represents the running bitwise AND of a contiguous group.
      # Initialize with -1, which in two's complement represents all bits set to 1.
      current_and = -1
      
      for num in nums:
        # If current_and is -1, we are starting a new group.
        if current_and == -1:
          current_and = num
        else:
          current_and &= num
        
        # Check if the current group's AND satisfies the submask condition.
        if (current_and | target_or_val) == target_or_val:
          # If it does, we can "finalize" this group and start a new one.
          # This group does not require any more internal operations.
          current_and = -1
        else:
          # If not, the current element MUST be merged with the next one.
          # This merge counts as one operation.
          ops_needed += 1
          
      return ops_needed <= k

    # Binary search for the minimum possible OR value.
    low = 0
    # The maximum possible value is less than 2^30.
    high = (1 << 30) - 1
    ans = high

    while low <= high:
      mid = (low + high) // 2
      
      if check(mid):
        # If `mid` is achievable, it's a potential answer. Try for a smaller one.
        ans = mid
        high = mid - 1
      else:
        # If `mid` is not achievable, we need to allow more bits to be 1.
        low = mid + 1
            
    return ans