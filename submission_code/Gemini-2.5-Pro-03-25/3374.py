from typing import List # Import List for type hinting

class Solution:
    """
    Solves the problem of counting alternating subarrays in a binary array.
    An alternating subarray is one where no two adjacent elements are the same.
    """
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        """
        Calculates the total number of alternating subarrays using a dynamic 
        programming approach with O(1) space. It iterates through the array, 
        maintaining the length of the current alternating subarray ending
        at the current index. The total count is the sum of these lengths.

        Args:
            nums: The input binary array (list of 0s and 1s). 
                  Constraints: 1 <= nums.length <= 10^5, nums[i] is 0 or 1.

        Returns:
            The total count of alternating subarrays.
        """
        n = len(nums)
        
        # According to constraints, n >= 1, but handling n=0 is good practice.
        if n == 0:
            return 0
        
        # Initialize the total count of alternating subarrays found so far.
        total_count = 0
        
        # Stores the length of the current alternating subarray ending at index i.
        # This variable will be updated inside the loop. We initialize it to 0
        # before the loop, but it effectively starts at 1 for the first element.
        current_length = 0 
        
        # Iterate through the array elements using their index `i`.
        for i in range(n):
            # Check if the current element continues the alternating pattern 
            # from the previous one. This requires two conditions:
            # 1. We are not at the first element (i > 0).
            # 2. The current element `nums[i]` is different from the previous `nums[i-1]`.
            if i > 0 and nums[i] != nums[i-1]:
                # If the pattern continues (e.g., ...0, 1 or ...1, 0), 
                # increment the length of the current alternating subarray.
                # The length increases by 1 because we are adding nums[i].
                current_length += 1
            else:
                # This 'else' block handles two cases:
                # 1. It's the first element (i == 0). The alternating subarray is just [nums[0]].
                # 2. The current element breaks the alternating pattern (nums[i] == nums[i-1], e.g., ...0, 0 or ...1, 1).
                # In both cases, a new alternating subarray starts at index `i`, 
                # and its length is initially 1 (containing only nums[i]).
                current_length = 1
            
            # The key insight is that the number of alternating subarrays ending *exactly* at 
            # index `i` is equal to `current_length`. 
            # For example, if the alternating subarray ending at `i` is [a, b, c] (length 3), 
            # then the alternating subarrays ending at `i` are [c], [b, c], and [a, b, c]. There are 3 such subarrays.
            # Add this number (`current_length`) to the overall total count.
            total_count += current_length
            
        # After iterating through all elements, total_count holds the sum of counts
        # for alternating subarrays ending at each index, which gives the overall total count.
        return total_count