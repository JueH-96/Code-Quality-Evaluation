from typing import List

class Solution:
  def countSubarrays(self, nums: List[int], k: int) -> int:
    n = len(nums)
    
    # Constraints: 1 <= nums.length, so nums is not empty.
    # Find the maximum element in the array.
    # max_val = 0
    # for x in nums:
    #     if x > max_val:
    #         max_val = x
    # A more Pythonic way, typically faster:
    max_val = max(nums)

    ans = 0
    left = 0
    max_val_count_in_window = 0
    
    # Iterate through the array with the right pointer of the sliding window.
    for right in range(n):
      # If the current element is the maximum element, increment its count in the window.
      if nums[right] == max_val:
        max_val_count_in_window += 1
      
      # While the number of maximum elements in the current window (nums[left...right])
      # is at least k:
      while max_val_count_in_window >= k:
        # The current window nums[left...right] satisfies the condition.
        # Any subarray that starts at the current `left` index and ends at `right`
        # or any subsequent index (i.e., nums[left...right], nums[left...right+1], ..., nums[left...n-1])
        # will also satisfy the condition (as they contain nums[left...right]).
        # There are (n - 1) - right + 1 = n - right such subarrays.
        # Add this count to the total answer.
        ans += (n - right)
        
        # We have now counted all valid subarrays starting at the current `left` index
        # that are found based on the current `right` endpoint.
        # To find other potential valid subarrays, we need to shrink the window from the left.
        # If the element at the `left` pointer is the maximum element, its removal
        # will decrease the count of maximum elements in the window.
        if nums[left] == max_val:
          max_val_count_in_window -= 1
        # Move the left pointer one step to the right.
        left += 1
        # The `while` loop will then re-evaluate the condition for the new, smaller window
        # nums[left...right]. If it's still valid, we repeat the counting process.
        
    return ans