import collections

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        """
        Calculates the length of the longest good subarray using a sliding window.

        A subarray is "good" if the frequency of each element is at most k.
        """
        # A hash map to store the frequency of elements in the current window.
        freq = collections.defaultdict(int)
        
        # The left pointer of the sliding window.
        left = 0
        
        # The result to store the maximum length of a good subarray found so far.
        max_length = 0
        
        # Iterate through the array with the right pointer to expand the window.
        for right, num in enumerate(nums):
            # Add the current element to the window and update its frequency.
            freq[num] += 1
            
            # If the frequency of the current element exceeds k, the window is invalid.
            # We must shrink the window from the left until it becomes valid again.
            while freq[num] > k:
                # Get the element at the left end of the window.
                left_num = nums[left]
                
                # Decrease its frequency as it's being removed from the window.
                freq[left_num] -= 1
                
                # Move the left pointer to the right, shrinking the window.
                left += 1
            
            # At this point, the window [left, right] is a "good" subarray.
            # We calculate its length and update the maximum length found so far.
            max_length = max(max_length, right - left + 1)
            
        return max_length