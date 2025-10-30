import collections
from typing import List # Required by the problem's starter code signature

class Solution:
  def maxSubarrayLength(self, nums: List[int], k: int) -> int:
    # freq will store the frequency of numbers in the current window [left, right]
    freq = collections.defaultdict(int)
    max_length = 0
    left = 0
    
    # Expand the window by moving the right pointer
    for right in range(len(nums)):
        current_num = nums[right]
        freq[current_num] += 1
        
        # If the frequency of current_num (the element just added at nums[right])
        # exceeds k, we need to shrink the window from the left.
        # We continue shrinking as long as freq[current_num] is too high.
        # This ensures that when the while loop terminates, freq[current_num] <= k.
        # Other elements x in the window must also have freq[x] <= k because:
        #  - They were already <= k before current_num was added and potentially violated the condition.
        #  - Their frequencies do not increase during the shrinking process (they either decrease if nums[left]=x, or stay same).
        while freq[current_num] > k:
            left_num = nums[left]
            freq[left_num] -= 1
            
            # If freq[left_num] becomes 0, it effectively means left_num is no longer
            # in the active window count. defaultdict handles this fine.
            # Explicitly deleting the key (e.g., if freq[left_num] == 0: del freq[left_num])
            # is a possible micro-optimization for memory but not necessary for correctness.
            
            left += 1
        
        # After the while loop, the window nums[left...right] is "good" 
        # (all element frequencies are <= k).
        # Calculate the length of this good subarray and update max_length.
        max_length = max(max_length, right - left + 1)
            
    return max_length