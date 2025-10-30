import collections
from typing import List

class Solution:
  def isPossibleToSplit(self, nums: List[int]) -> bool:
    """
    Determines if an array can be split into two sub-arrays nums1 and nums2
    of equal length, where both nums1 and nums2 contain distinct elements.
    """
    
    # The problem constraints state:
    # nums1.length == nums2.length == nums.length / 2
    # nums1 should contain distinct elements.
    # nums2 should also contain distinct elements.
    
    # Core logic:
    # For nums1 to contain distinct elements, any specific number can appear at most once in nums1.
    # Similarly, for nums2 to contain distinct elements, any specific number can appear at most once in nums2.
    #
    # Consider any number `x` present in the original array `nums`.
    # If `x` is to be part of the split:
    # - It can be placed in nums1 (at most once, due to distinctness).
    # - It can be placed in nums2 (at most once, due to distinctness).
    # This means that across both nums1 and nums2, the number `x` can appear at most twice.
    # Since nums1 and nums2 together must form the original multiset `nums` (i.e., contain all elements
    # of nums with their original multiplicities), the frequency of `x` in `nums` cannot exceed 2.
    #
    # If any number appears 3 or more times in `nums`, it's impossible to satisfy
    # the distinctness condition for both nums1 and nums2. For example, if `x` appears
    # 3 times:
    #   - We could place one `x` in nums1.
    #   - We could place one `x` in nums2.
    #   - The third `x` cannot be placed in nums1 (as it would make nums1 contain `x` twice)
    #     nor in nums2 (as it would make nums2 contain `x` twice), without violating distinctness.
    #
    # Therefore, a necessary condition for a valid split is that every number in `nums`
    # must have a frequency of at most 2.
    
    # This condition is also sufficient, as shown by a constructive argument (detailed in thought process).
    # In summary, elements appearing twice are split one to each sub-array. Elements appearing once
    # are used to fill the remaining capacity in each sub-array. The counts ensure this is always possible
    # if no element appears more than twice.
    
    # Practical Algorithm:
    # 1. Calculate frequencies of all numbers in `nums`. `collections.Counter` is suitable for this.
    # 2. Iterate through the calculated frequencies. If any frequency is greater than 2, return `False`.
    # 3. If the loop completes without finding such a frequency, it means all frequencies are
    #    2 or less. In this case, return `True`.
    
    counts = collections.Counter(nums)
    
    # Check the frequency of each distinct number found in nums
    for freq in counts.values(): # .values() gives an iterable of the frequencies
        if freq > 2:
            return False  # Found a number that appears more than twice
            
    # If we reach here, all numbers appear at most twice
    return True