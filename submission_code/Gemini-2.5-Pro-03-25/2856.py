import collections
from typing import List # Make sure List is imported if not implicitly available

class Solution:
    """
    This class provides a solution to count complete subarrays.
    A complete subarray has the same number of distinct elements as the entire array.
    """
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        """
        Counts the number of complete subarrays in the input array nums using an efficient O(N) sliding window approach.

        Args:
            nums: A list of positive integers.

        Returns:
            The total number of complete subarrays.
        """
        n = len(nums)
        
        # Step 1: Find the number of distinct elements in the whole array.
        # This value 'k' is the target number of distinct elements for any subarray to be considered 'complete'.
        k = len(set(nums))
        
        complete_count = 0  # Initialize the counter for complete subarrays.
        left = 0            # Initialize the left pointer of the sliding window.
        
        # Use collections.Counter to efficiently store and update the frequency of elements 
        # within the current sliding window defined by [left, right].
        counts = collections.Counter() 
        
        # Step 2: Iterate through the array using the 'right' pointer to expand the window.
        for right in range(n):
            # Add the element nums[right] to the current window and update its frequency.
            counts[nums[right]] += 1
            
            # Step 3: Check if the current window [left, right] has exactly 'k' distinct elements.
            # The number of distinct elements in the window is given by len(counts).
            # If the window is complete (len(counts) == k), we need to count how many subarrays this implies
            # and then try to shrink the window from the left while maintaining completeness.
            while len(counts) == k:
                # If the window nums[left..right] is complete (contains k distinct elements), 
                # then any subarray that starts at index 'left' and ends at index 'j', where j >= right, 
                # must also contain at least these k distinct elements. Since k is the total number of 
                # distinct elements in the entire array, these subarrays must also have exactly k distinct elements.
                # The subarrays are: nums[left..right], nums[left..right+1], ..., nums[left..n-1].
                # The number of such subarrays starting at 'left' is (n - 1) - right + 1 = n - right.
                # Add this count to our total.
                complete_count += n - right
                
                # Now, attempt to shrink the window from the left side.
                # Decrease the frequency count of the element at the current left pointer.
                counts[nums[left]] -= 1
                
                # If the frequency of nums[left] becomes 0 after decrementing, it means this 
                # element is no longer present in the window starting from left+1. 
                # We must remove this element key from the counter to accurately reflect the 
                # number of distinct elements in the potentially smaller window.
                if counts[nums[left]] == 0:
                    del counts[nums[left]]
                
                # Move the left pointer one step to the right, effectively shrinking the window.
                left += 1
                
                # The 'while' loop condition (len(counts) == k) will re-evaluate if the newly formed 
                # window [left..right] is still complete. If it is, we continue the process: 
                # count the subarrays and shrink further from the left.
                
        # After the loop finishes (right pointer reaches the end of the array),
        # complete_count will hold the total number of complete subarrays.
        return complete_count