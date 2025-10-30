import collections
from typing import List

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        
        # Initialize variables for the sliding window
        # max_sum will store the maximum sum found so far for almost unique subarrays.
        # It's initialized to 0, which is the required return value if no such subarray exists.
        max_sum = 0
        
        # current_sum will store the sum of elements in the current window.
        current_sum = 0 
        
        # window_counts will store the frequency of elements in the current window.
        # Using collections.Counter is efficient for managing element counts and determining
        # the number of distinct elements (via len(window_counts)).
        window_counts = collections.Counter()
        
        # Step 1: Process the first window of size k
        # This loop initializes current_sum and window_counts for the first subarray
        # (elements from index 0 to k-1).
        for i in range(k):
            current_sum += nums[i]
            window_counts[nums[i]] += 1
        
        # After forming the first window, check if it is an "almost unique" subarray.
        # An "almost unique" subarray must have at least 'm' distinct elements.
        if len(window_counts) >= m:
            max_sum = current_sum
            
        # Step 2: Slide the window across the rest of the array
        # The loop starts from index 'k' because 'k' will be the rightmost index
        # of the second window (nums[1] to nums[k]).
        for i in range(k, n):
            # Remove the element that is leaving the window from the left.
            # This element is at index (i - k).
            left_element = nums[i - k]
            current_sum -= left_element
            window_counts[left_element] -= 1
            
            # If the count of left_element becomes 0, it means this element is no longer
            # present in the current window. We delete it from window_counts to ensure
            # that len(window_counts) accurately reflects only the distinct elements
            # currently within the window.
            if window_counts[left_element] == 0:
                del window_counts[left_element]
            
            # Add the new element that is entering the window from the right.
            # This new element is at the current index 'i'.
            right_element = nums[i]
            current_sum += right_element
            window_counts[right_element] += 1
            
            # After updating the window (by removing one element and adding one),
            # check if this new window is "almost unique".
            if len(window_counts) >= m:
                # If it is, update max_sum if the current_sum is greater than the
                # previously recorded maximum sum.
                max_sum = max(max_sum, current_sum)
                
        # Return the maximum sum found. If no "almost unique" subarray of length k was found,
        # max_sum will remain 0, which is the specified return value for that case.
        return max_sum