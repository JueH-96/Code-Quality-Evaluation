from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        """
        Finds the minimum absolute difference between k and the bitwise OR
        of any subarray elements.

        The approach iterates through the array, maintaining the set of distinct
        bitwise OR values for all subarrays ending at the current index.
        For each new element nums[i], the new set of OR values ending at index i
        is formed by taking nums[i] itself, and for each OR value v from subarrays
        ending at index i-1, calculating v | nums[i].
        Since the bitwise OR operation is monotonic (bits only turn on), the number
        of distinct OR values ending at any index is limited by the number of bits
        in the maximum possible value (around 30 for 10^9).

        Args:
            nums: A list of integers.
            k: An integer target value.

        Returns:
            The minimum absolute difference |k - OR_value|.
        """
        # Initialize the minimum difference to a very large value.
        min_diff = float('inf')
        
        # This set will store the distinct bitwise OR values of all subarrays
        # ending at the previous index (i-1).
        # Initially, there are no subarrays ending before the first element.
        prev_ors = set() 
        
        # Iterate through each number in the input array.
        for num in nums:
            # current_ors will store the distinct bitwise OR values of all subarrays
            # ending at the current number `num` (which is nums[i]).
            current_ors = set()
            
            # The current number itself forms a subarray [num] ending at index i.
            # Its OR value is just the number itself.
            current_ors.add(num)
            
            # For every distinct OR value `prev_or_val` found from subarrays ending
            # at the previous index (i-1), we can form a new subarray ending
            # at the current index `i` by extending it with `num`.
            # The OR value of this new subarray is `prev_or_val | num`.
            for prev_or_val in prev_ors:
                current_ors.add(prev_or_val | num)
            
            # After computing all distinct OR values ending at the current index `i`,
            # iterate through them and update the minimum absolute difference found so far.
            for or_val in current_ors:
                min_diff = min(min_diff, abs(k - or_val))
            
            # The set of distinct OR values ending at the current index `i` becomes
            # the 'prev_ors' set for the next iteration (index i+1).
            prev_ors = current_ors
        
        # Return the minimum difference found across all subarrays.
        return min_diff