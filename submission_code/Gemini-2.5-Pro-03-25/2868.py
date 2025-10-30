import collections
from typing import List

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        """
        Counts the total number of continuous subarrays in the given list of numbers using a sliding window approach
        with monotonic deques. A subarray nums[i..j] is continuous if for any two elements nums[i1], nums[i2]
        within the subarray (i <= i1, i2 <= j), the absolute difference |nums[i1] - nums[i2]| is at most 2.
        This condition is equivalent to checking if max(subarray) - min(subarray) <= 2.

        Args:
          nums: A list of integers.

        Returns:
          The total count of continuous subarrays. An integer.
        """
        N = len(nums)
        # Deque to maintain indices of candidates for maximum element in the sliding window [left, right].
        # Stores indices `k` such that `nums[k]` values are in decreasing order.
        # The front of the deque always holds the index of the maximum element in the current window.
        max_deque = collections.deque() 
        
        # Deque to maintain indices of candidates for minimum element in the sliding window [left, right].
        # Stores indices `k` such that `nums[k]` values are in increasing order.
        # The front of the deque always holds the index of the minimum element in the current window.
        min_deque = collections.deque() 
        
        left = 0  # The left pointer (start index) of the sliding window [left, right]
        total_count = 0 # Accumulator for the count of continuous subarrays. Initialized to 0.
        
        # Iterate through the array with the right pointer `right` from 0 to N-1
        for right in range(N):
            # Update max_deque to include the current element nums[right].
            # To maintain the decreasing property, remove indices from the back corresponding to elements 
            # that are less than or equal to nums[right]. These elements cannot be the maximum anymore
            # if nums[right] is included and they appear before nums[right].
            while max_deque and nums[max_deque[-1]] <= nums[right]:
                max_deque.pop()
            # Add the current index `right` to the max_deque.
            max_deque.append(right)
            
            # Update min_deque similarly to include nums[right] while maintaining the increasing property.
            # Remove indices from the back corresponding to elements that are greater than or equal to nums[right].
            while min_deque and nums[min_deque[-1]] >= nums[right]:
                min_deque.pop()
            # Add the current index `right` to the min_deque.
            min_deque.append(right)
            
            # Check if the current window [left, right] violates the condition: max(window) - min(window) <= 2.
            # The maximum value in the window is nums[max_deque[0]].
            # The minimum value in the window is nums[min_deque[0]].
            # Ensure both deques are non-empty before accessing their front elements.
            # While the condition is violated (difference > 2):
            while max_deque and min_deque and nums[max_deque[0]] - nums[min_deque[0]] > 2:
                # The window is invalid. We must shrink it from the left by incrementing `left`.
                
                # The element at index `left` is about to be removed from the window's scope.
                # Check if this element is responsible for the current maximum value (i.e., its index is at the front of max_deque).
                # If so, remove its index from max_deque's front because it's no longer part of the window [left+1, right].
                if max_deque[0] == left:
                    max_deque.popleft()
                    
                # Similarly, check if the element at index `left` is responsible for the current minimum value.
                # If so, remove its index from min_deque's front.
                if min_deque[0] == left:
                    min_deque.popleft()
                
                # Move the left boundary of the window one step to the right.
                left += 1

            # After the while loop, the window [left, right] is guaranteed to satisfy the condition:
            # max(nums[left..right]) - min(nums[left..right]) <= 2.
            # Consequently, all subarrays ending at `right` and starting at an index `k` such that `left <= k <= right`
            # are also continuous subarrays.
            # The number of such valid subarrays ending at `right` is `right - left + 1`.
            # Add this number to the total count.
            total_count += (right - left + 1)
                
        # After iterating through all possible right endpoints, return the total accumulated count.
        return total_count