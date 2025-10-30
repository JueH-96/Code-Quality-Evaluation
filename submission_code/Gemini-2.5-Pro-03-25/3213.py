from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        """
        Counts the number of subarrays where the maximum element of nums 
        appears at least k times.

        Args:
            nums: A list of integers.
            k: A positive integer.

        Returns:
            The number of valid subarrays.
        
        Approach:
        This problem can be solved efficiently using a sliding window approach.
        The core idea is to iterate through the array with a `right` pointer,
        expanding a window `[left, right]`. We maintain a count of the maximum
        element (`max_val`) within this window.

        1. Find the maximum element (`max_val`) in the input array `nums`.
        2. Initialize `left = 0`, `count = 0` (count of `max_val` in the window),
           and `result = 0` (total count of valid subarrays).
        3. Iterate `right` from `0` to `n-1` (where `n` is the length of `nums`):
           a. If `nums[right]` is equal to `max_val`, increment `count`.
           b. While `count` is greater than or equal to `k`:
              This indicates that the current window `[left, right]` might contain
              too many `max_val` elements, or it's just become valid. We need to
              shrink the window from the left until the count of `max_val` is
              strictly less than `k`.
              - If `nums[left]` is `max_val`, decrement `count` because this
                element is about to be removed from the window.
              - Increment `left`.
           c. After the `while` loop, `left` points to the smallest index such
              that the window `nums[left:right+1]` contains *fewer* than `k`
              occurrences of `max_val`. This implies that any subarray ending
              at the current `right` index and starting at an index `i` where
              `0 <= i < left` must contain at least `k` occurrences of `max_val`.
              The number of such valid starting indices is exactly `left`.
              Therefore, add `left` to the `result`.
        4. Return the final `result`.

        Time Complexity: O(N), where N is the length of `nums`. We iterate through
                         the array once with the `right` pointer, and the `left`
                         pointer also moves at most N steps. Finding the maximum
                         element takes O(N).
        Space Complexity: O(1), as we only use a few variables for pointers and counts.
        """
        
        n = len(nums)
        # Constraints state 1 <= nums.length <= 10^5, so n >= 1.
        # If the constraints allowed an empty list, we would handle it here.
            
        # Find the maximum element in nums.
        # Using the built-in max() function is efficient.
        # Constraints ensure nums is not empty.
        max_val = max(nums)

        # Initialize the left pointer of the sliding window.
        left = 0
        # Initialize the count of the maximum element within the current window [left, right].
        count = 0  
        # Initialize the total count of valid subarrays.
        result = 0

        # Iterate through the array with the right pointer of the sliding window.
        for right in range(n):
            # If the current element is the maximum value, increment the count.
            if nums[right] == max_val:
                count += 1
            
            # While the current window [left, right] has k or more occurrences
            # of the maximum element, shrink the window from the left.
            # This loop advances `left` until the window `nums[left:right+1]`
            # contains strictly less than k maximum elements.
            while count >= k:
                # If the element at the left pointer is the maximum value,
                # decrement the count as it's being removed from the window.
                if nums[left] == max_val:
                    count -= 1
                # Move the left pointer to the right, effectively shrinking the window.
                left += 1
                
            # At this point, `left` is the smallest index such that the subarray
            # `nums[left:right+1]` contains fewer than k maximum elements.
            # This implies that any subarray ending at `right` and starting at an index `i`
            # where `0 <= i < left` must contain at least k maximum elements.
            # These are the valid subarrays ending at the current `right`.
            # The number of such valid starting indices is `left`.
            # Add this number (the count of valid subarrays ending at `right`) 
            # to the total result.
            result += left
            
        # Return the final count of valid subarrays found.
        return result