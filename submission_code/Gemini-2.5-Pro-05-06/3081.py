from typing import List

class Solution:
  def minLengthAfterRemovals(self, nums: List[int]) -> int:
    n = len(nums)
    
    # Constraints state 1 <= nums.length, so n is at least 1.
    # If n is 0 (e.g. if constraints were different), result is 0.
    # if n == 0:
    #   return 0

    # Two-pointer greedy strategy:
    # Pointer i for the first half: nums[0 ... n//2 - 1]
    # Pointer j for the second half: nums[n//2 ... n - 1]
    
    i = 0
    # j starts at the beginning of the second conceptual half.
    # For n=5, n//2 = 2. First half indices [0,1]. Second half indices [2,3,4]. j starts at 2.
    # Number of elements in the first half is n // 2.
    # Number of elements in the second half is n - (n // 2).
    j = n // 2 
    
    pairs_removed = 0
    
    # Loop while pointer i is in the first half and pointer j is in the second half.
    # The first half is nums[0...n//2-1]. So i goes from 0 up to (n//2)-1.
    # The second half is nums[n//2...n-1]. So j goes from n//2 up to n-1.
    while i < (n // 2) and j < n:
      if nums[i] < nums[j]:
        # Found a pair (nums[i], nums[j]) that can be removed.
        # The condition i < j for removal is satisfied because elements for nums[i]
        # are from original indices < n//2, and for nums[j] from original indices >= n//2.
        pairs_removed += 1
        i += 1 # Consume nums[i]
        j += 1 # Consume nums[j]
      else: # nums[i] >= nums[j]
        # nums[i] cannot be paired with nums[j].
        # Since nums is sorted, nums[j] is too small for nums[i].
        # nums[j] will also be too small for any subsequent element nums[k] (k > i) from the first half,
        # because nums[k] >= nums[i].
        # So, advance j to find a larger element in the second half to pair with the current nums[i].
        j += 1
        
    # Each pair removal reduces the length by 2.
    return n - (2 * pairs_removed)