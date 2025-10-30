from typing import List

class Solution:
  def minimumRightShifts(self, nums: List[int]) -> int:
    n = len(nums)
    
    # Constraints: 1 <= nums.length <= 100.
    # So n is at least 1.
    
    if n == 1:
      # An array with one element is always sorted.
      # 0 shifts are needed.
      return 0

    dip_count = 0
    # dip_idx will store the index i where nums[i] > nums[(i + 1) % n].
    # If there's only one such dip, this index is what we need.
    dip_idx = -1 

    for i in range(n):
      # Compare current element with the next element (cyclically)
      if nums[i] > nums[(i + 1) % n]:
        dip_count += 1
        dip_idx = i
    
    # After checking all elements:
    #
    # For n > 1 and distinct elements, dip_count must be at least 1.
    # If dip_count were 0, it would imply nums[0] < nums[1] < ... < nums[n-1] < nums[0] 
    # (since elements are distinct). This forms a contradiction (e.g., x < y < z < x is impossible).
    # The n=1 case (where dip_count can be 0 because nums[0] > nums[0] is false) 
    # is handled by the initial check.
    
    if dip_count == 1:
      # If there is exactly one dip, the array is a rotation of a sorted array.
      # The dip occurs at index `dip_idx`.
      # This means nums[0...dip_idx] are the elements that "wrap around" from the end of the sorted sequence.
      # The number of elements in this prefix is `dip_idx + 1`.
      # These (dip_idx + 1) elements effectively need to be "left-shifted" out of the way,
      # or equivalently, the remaining n - (dip_idx + 1) elements from the end need to be
      # right-shifted to the front.
      #
      # The number of right shifts needed is (n - (number of elements in the prefix)) % n.
      # So, shifts = (n - (dip_idx + 1)) % n.
      #
      # Example: nums = [1,2,3,4,5] (already sorted)
      # n=5. The only dip is nums[4]=5 > nums[0]=1. So, dip_idx = 4.
      # Shifts = (5 - (4+1)) % 5 = (5 - 5) % 5 = 0. Correct.
      #
      # Example: nums = [3,4,5,1,2]
      # n=5. The dip is nums[2]=5 > nums[3]=1. So, dip_idx = 2.
      # Shifts = (5 - (2+1)) % 5 = (5 - 3) % 5 = 2. Correct.
      
      return (n - (dip_idx + 1)) % n
    else: # dip_count > 1 (or dip_count == 0, which is impossible for n > 1 as explained above)
      # If dip_count > 1, the array has multiple points of disorder
      # and cannot be sorted by simple right shifts.
      # Example: [2,1,4]. Dips: 2>1 (at i=0) and 4>2 (at i=2, cyclically). dip_count=2.
      return -1