from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        """
        Calculates the maximum possible bitwise OR of an array after applying
        a multiplication-by-2 operation at most k times.
        """
        n = len(nums)

        # suffix_or[i] will store the bitwise OR of nums[i], nums[i+1], ..., nums[n-1].
        # The size is n+1 to handle the edge case for the last element gracefully.
        # suffix_or[n] will be 0.
        suffix_or = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_or[i] = suffix_or[i + 1] | nums[i]

        max_or_value = 0
        
        # left_or will store the running bitwise OR of elements from the beginning
        # up to the element just before the current one.
        left_or = 0
        
        # It's always optimal to apply all k operations to a single number.
        # We iterate through each number, assume all k operations are applied to it,
        # and calculate the resulting total OR.
        for i in range(n):
            # The OR of all elements to the right of nums[i].
            right_or = suffix_or[i + 1]
            
            # The OR of all elements except nums[i].
            others_or = left_or | right_or
            
            # The value of nums[i] after k operations (multiplied by 2^k).
            boosted_num = nums[i] << k
            
            # The total OR if nums[i] is the one we boost.
            current_or_value = boosted_num | others_or
            
            max_or_value = max(max_or_value, current_or_value)
            
            # Update left_or for the next iteration to include the current number.
            left_or |= nums[i]

        return max_or_value