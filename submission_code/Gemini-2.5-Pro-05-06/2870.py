from typing import List

class Solution:
  def alternatingSubarray(self, nums: List[int]) -> int:
    n = len(nums)
    # Constraints: 2 <= nums.length <= 100. So n is guaranteed to be at least 2.
    # "m is greater than 1." This means any valid alternating subarray must have length at least 2.

    max_len = -1 # Initialize with -1, as per problem if no such subarray exists.

    # Iterate through all possible starting positions 'i' for a subarray.
    # The subarray starts with s_0 = nums[i] and s_1 = nums[i+1].
    # So, 'i' can go up to n-2 (i.e., nums[n-2] and nums[n-1] as the first pair).
    for i in range(n - 1):
        # Check the first condition for an alternating subarray: s_1 = s_0 + 1
        # Here, s_0 corresponds to nums[i], and s_1 to nums[i+1].
        if nums[i+1] - nums[i] == 1:
            # If the condition holds, we have found an alternating subarray of at least length 2:
            # [nums[i], nums[i+1]]
            current_len = 2
            max_len = max(max_len, current_len) # Update max_len
            
            # Now, try to extend this alternating subarray.
            # The pattern is s_0, s_1, s_0, s_1, ...
            # This means s_k should be nums[i] (the s_0 value) if k is even, 
            # and nums[i+1] (the s_1 value) if k is odd (0-indexed k for subarray elements).
            
            # k_in_nums is the index in the original 'nums' array for subsequent elements.
            # It starts from i+2, which corresponds to s_2 in the subarray.
            for k_in_nums in range(i + 2, n):
                # idx_in_s is the 0-based index within the current subarray s that starts at nums[i].
                # s_0 is nums[i], s_1 is nums[i+1], s_2 is nums[i+2], ...
                # So, for nums[k_in_nums], its index in the subarray is (k_in_nums - i).
                idx_in_s = k_in_nums - i
                
                # Determine the expected value for nums[k_in_nums] based on the alternating pattern:
                # If idx_in_s is even (e.g., for s_2, s_4), expected value is nums[i] (s_0).
                # If idx_in_s is odd (e.g., for s_3, s_5), expected value is nums[i+1] (s_1).
                # This can be expressed as: expected_value = nums[i + (idx_in_s % 2)]
                
                expected_value_at_current_position = nums[i + (idx_in_s % 2)]
                
                if nums[k_in_nums] == expected_value_at_current_position:
                    # The pattern continues, extend the current alternating subarray.
                    current_len += 1
                    max_len = max(max_len, current_len) # Update max_len
                else:
                    # The pattern is broken. This subarray cannot be extended further.
                    # Break from the inner loop and try the next starting position 'i'.
                    break 
        # If nums[i+1] - nums[i] != 1, then nums[i] cannot be the start of an
        # alternating subarray as per the definition s_1 = s_0 + 1.
        # The outer loop will continue to the next 'i'.

    return max_len