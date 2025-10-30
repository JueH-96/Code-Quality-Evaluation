from typing import List

class Solution:
  def minOperations(self, nums: List[int]) -> int:
    n = len(nums)
    operations = 0

    # Constraints: 3 <= nums.length <= 10^5. So n >= 3.
    
    # Iterate from i = 0 up to n-3.
    # range(n-2) will produce integers 0, 1, ..., n-3.
    for i in range(n - 2):
      if nums[i] == 0:
        # If nums[i] is 0, we must perform an operation starting at index i.
        # This is the only way to change nums[i] to 1 without affecting
        # nums[0]...nums[i-1] which are already processed and fixed.
        operations += 1
        
        # Perform the flip operation on nums[i], nums[i+1], nums[i+2].
        # nums[i] was 0, so it becomes 1.
        nums[i] = 1 
        # nums[i+1] and nums[i+2] are flipped (0 to 1, or 1 to 0).
        nums[i+1] = 1 - nums[i+1]
        nums[i+2] = 1 - nums[i+2]
      # If nums[i] is already 1, we do nothing. It's in the target state.
      # An operation starting at i would flip it to 0, which is not desired.

    # After the loop, elements nums[0]...nums[n-3] have been processed.
    # The element nums[i] (for i <= n-3) is guaranteed to be 1 after its step.
    # The states of nums[n-2] and nums[n-1] are determined by previous operations.
    # The last operation that could affect them is op(n-3), which flips 
    # (nums[n-3], nums[n-2], nums[n-1]).
    #
    # We need to check if the last two elements, nums[n-2] and nums[n-1], are 1.
    # nums[n-3] is already guaranteed to be 1 due to the processing at i = n-3.
    # If nums[n-2] and nums[n-1] are 1, then all elements are 1.
    if nums[n-2] == 1 and nums[n-1] == 1:
      return operations
    else:
      # If either nums[n-2] or nums[n-1] (or both) is 0, it's impossible.
      return -1