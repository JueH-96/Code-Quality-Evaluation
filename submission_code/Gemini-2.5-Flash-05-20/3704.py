from typing import List

class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Calculate the total sum of all elements in the array.
        total_sum = sum(nums)
        
        # A partition is valid if the difference between the sum of the left
        # and right subarrays (S_L - S_R) is even.
        # This condition is equivalent to S_L and S_R having the same parity.
        # (i.e., both even or both odd).
        
        # We know that S_total = S_L + S_R.
        #
        # If S_total is even:
        #   S_L + S_R = even. This implies (S_L is even and S_R is even) OR
        #   (S_L is odd and S_R is odd). In both cases, S_L and S_R have the same parity.
        #   Thus, (S_L - S_R) will always be even.
        #   Therefore, if the total sum is even, all possible partitions are valid.
        #
        # If S_total is odd:
        #   S_L + S_R = odd. This implies (S_L is even and S_R is odd) OR
        #   (S_L is odd and S_R is even). In both cases, S_L and S_R have different parities.
        #   Thus, (S_L - S_R) will always be odd.
        #   Therefore, if the total sum is odd, no possible partition is valid.
        
        # The number of possible partition points 'i' (where 0 <= i < n - 1)
        # ranges from 0 to n-2. This means there are (n - 2) - 0 + 1 = n - 1 total partitions.
        
        if total_sum % 2 == 0:
            # If the total sum is even, all n-1 possible partitions are valid.
            return n - 1
        else:
            # If the total sum is odd, no partition will result in an even difference.
            return 0