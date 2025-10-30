import collections
from typing import List

# Starter code provided by the user
class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        """
        Finds the length of the longest good subarray using a sliding window.

        A subarray is good if the frequency of each element is less than or equal to k.
        The sliding window approach maintains a window [left, right] and expands it
        to the right. If the window becomes "bad" (violates the frequency constraint),
        it shrinks from the left until it becomes "good" again. The maximum length
        of a good window encountered is tracked.

        Args:
            nums: The input list of integers. Constraints: 1 <= nums.length <= 10^5, 1 <= nums[i] <= 10^9.
            k: The maximum allowed frequency for each element in a good subarray. Constraints: 1 <= k <= nums.length.

        Returns:
            The length of the longest good subarray.
        """
        # Get the length of the input array. Constraints state 1 <= n.
        n = len(nums)
        
        # Initialize a dictionary to store frequencies of elements in the current window [left, right].
        # collections.defaultdict(int) automatically initializes frequency to 0 for new keys,
        # simplifying frequency updates.
        freq = collections.defaultdict(int)
        
        # Initialize the left pointer of the sliding window.
        left = 0
        # Initialize the variable to store the maximum length of a good subarray found so far.
        max_length = 0
        
        # Iterate through the array using the right pointer of the sliding window.
        for right in range(n):
            # Get the element at the current right pointer.
            num_right = nums[right]

            # --- Expand the window by including the element at 'right' ---
            # Increment the frequency count for this element in the current window.
            freq[num_right] += 1

            # --- Shrink the window from the left if necessary ---
            # Check if the frequency of the newly added element (or any element currently
            # in the window) exceeds the allowed limit k. If freq[num_right] is now greater
            # than k, the window is no longer "good", and we must shrink it from the left
            # until the condition freq[num_right] <= k is met again.
            # Note: We only need to check the element just added (num_right). If increasing its count
            # made it exceed k, we must remove elements from the left until num_right's count drops back to k.
            # If another element num_other already exceeded k, the loop for num_other would have already
            # run in a previous iteration of `right` and shrunk the window appropriately.
            # However, consider the case where shrinking removes an element `x` which brings `freq[num_right]`
            # back to `k`, but `freq[x]` might still be `> k` if `x == num_left`.
            # The most robust approach is to shrink whenever *any* element's frequency is > k.
            # We check the condition `freq[num_right] > k` as the trigger, but the shrinking
            # continues until the window becomes valid for *all* elements. The condition inside
            # the while loop ensures this.

            # While the frequency of the element at the right pointer exceeds k,
            # we need to shrink the window from the left.
            while freq[num_right] > k:
                # Get the element at the current left pointer, which is about to be removed.
                num_left = nums[left]
                
                # Decrement the frequency count for the element at 'left' as it's
                # being removed from the window's frequency map.
                freq[num_left] -= 1
                
                # Move the left boundary of the window one step to the right.
                left += 1
            
            # The simpler condition `while freq[num_right] > k` is sufficient because:
            # 1. If adding `num_right` does not violate k for `num_right`, the window remains good
            #    (assuming it was good before), as no other frequency changes. The length increases.
            # 2. If adding `num_right` makes `freq[num_right] > k`, the window becomes bad.
            #    The `while` loop shrinks the window by incrementing `left`.
            #    This continues until `freq[num_right]` is no longer greater than `k`.
            #    Crucially, removing `nums[left]` can only decrease frequencies, so it cannot
            #    make another element `x` violate the `k` limit if it didn't already.
            #    Therefore, once `freq[num_right] <= k`, the window is guaranteed to be good again,
            #    as `num_right` was the only element whose frequency increased in this step.

            # --- Update the maximum length ---
            # After potentially shrinking, the window [left, right] is guaranteed 
            # to be "good" (all element frequencies are <= k).
            # Calculate the length of the current good window.
            current_length = right - left + 1
            # Update the overall maximum length found so far.
            max_length = max(max_length, current_length)

        # Return the maximum length found after iterating through the entire array.
        return max_length