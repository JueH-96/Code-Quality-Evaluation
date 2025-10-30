from typing import List

class Solution:
  def countAlternatingSubarrays(self, nums: List[int]) -> int:
    n = len(nums)
    
    # Constraints: 1 <= nums.length <= 10^5. So, n is always at least 1.
    # An array with one element [x] has one alternating subarray: [x] itself.
    # For n=1, the loop range(1,1) is empty.
    # total_alternating_subarrays initialized to 1 handles this base case correctly.
    
    # Initialize with the first element's contribution:
    # The subarray [nums[0]] is alternating and has length 1.
    # It contributes 1 to the total count.
    # The length of the alternating subarray ending at index 0 is 1.
    total_alternating_subarrays = 1
    current_streak_length = 1
    
    # Iterate from the second element (index 1) up to n-1.
    for i in range(1, n):
        if nums[i] != nums[i-1]:
            # If nums[i] is different from nums[i-1], the alternating
            # sequence ending at nums[i-1] can be extended by nums[i].
            # So, the length of the alternating sequence ending at nums[i] increases by 1.
            current_streak_length += 1
        else:
            # If nums[i] is the same as nums[i-1], the alternating sequence is broken.
            # The only alternating subarray ending at nums[i] is [nums[i]] itself.
            # Its length is 1.
            current_streak_length = 1
            
        # The value `current_streak_length` is the number of alternating subarrays
        # that end at index `i`. For example, if current_streak_length is k,
        # meaning nums[i-k+1 ... i] is alternating, then the alternating
        # subarrays ending at i are:
        #   [nums[i]]
        #   [nums[i-1], nums[i]]
        #   ...
        #   [nums[i-k+1], ..., nums[i]]
        # There are `current_streak_length` (which is k) such subarrays. We add this to the total.
        total_alternating_subarrays += current_streak_length
            
    return total_alternating_subarrays