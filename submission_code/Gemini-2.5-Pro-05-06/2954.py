from typing import List

class Solution:
  def maxSum(self, nums: List[int], m: int, k: int) -> int:
    n = len(nums)
    
    # Constraints:
    # 1 <= nums.length (so n >= 1)
    # 1 <= m <= k <= nums.length (so k >= 1, and k <= n)
    # These guarantees mean we don't need to handle edge cases like k > n or empty nums
    # within this core logic.

    max_achieved_sum = 0  # Stores the maximum sum found so far for an almost unique subarray
    current_window_sum = 0
    # window_element_counts will store frequencies of elements in the current window nums[left_ptr...right_ptr]
    window_element_counts = {} 

    left_ptr = 0
    # The right_ptr expands the window by including nums[right_ptr]
    for right_ptr in range(n):
        # Element entering the window
        element_in = nums[right_ptr]
        current_window_sum += element_in
        
        # Update frequency for element_in
        window_element_counts[element_in] = window_element_counts.get(element_in, 0) + 1

        # Check if the window has reached size k
        # The current window is nums[left_ptr...right_ptr]
        # Its size is (right_ptr - left_ptr + 1)
        if (right_ptr - left_ptr + 1) == k:
            # Window is now of size k. Evaluate it.
            # An almost unique subarray has at least m distinct elements.
            # The number of distinct elements in the window is len(window_element_counts).
            if len(window_element_counts) >= m:
                max_achieved_sum = max(max_achieved_sum, current_window_sum)
            
            # Prepare to slide the window: nums[left_ptr] will be removed.
            element_out = nums[left_ptr]
            current_window_sum -= element_out
            
            # Update frequency for element_out
            window_element_counts[element_out] -= 1
            if window_element_counts[element_out] == 0:
                # Remove the element from the map if its count drops to zero.
                # This is crucial for len(window_element_counts) to accurately reflect
                # the number of distinct elements in the window.
                del window_element_counts[element_out]
            
            left_ptr += 1 # Slide the window to the right
            
    return max_achieved_sum