from typing import List

class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Calculate the total sum of elements in the array
        total_sum = sum(nums)

        # The difference (sum_left - sum_right) is even if and only if 
        # sum_left and sum_right have the same parity.
        # (e.g., even - even = even, odd - odd = even)
        #
        # Also, sum_left + sum_right = total_sum.
        #
        # Consider the parity of total_sum:
        #
        # If sum_left and sum_right have the same parity, their sum (total_sum) must be even:
        #   - If sum_left is even and sum_right is even, total_sum = even + even = even.
        #   - If sum_left is odd and sum_right is odd, total_sum = odd + odd = even.
        #
        # This implies:
        # 1. If total_sum is odd:
        #    It's impossible for sum_left and sum_right to have the same parity.
        #    Therefore, (sum_left - sum_right) will always be odd.
        #    In this scenario, the number of partitions satisfying the condition is 0.
        
        if total_sum % 2 != 0:  # total_sum is odd
            return 0
        else:  # total_sum is even
            # 2. If total_sum is even:
            #    For any partition, sum_left + sum_right = total_sum (which is even).
            #    This means sum_left and sum_right must have the same parity.
            #    (Proof: If they had different parities, their sum would be odd, but total_sum is even. Contradiction.)
            #    - If sum_left is even, sum_right = total_sum (even) - sum_left (even) = even. (Same parity)
            #    - If sum_left is odd, sum_right = total_sum (even) - sum_left (odd) = odd. (Same parity)
            #    Since sum_left and sum_right always have the same parity (if total_sum is even),
            #    (sum_left - sum_right) will always be even for any partition.
            
            # The number of possible partitions is n - 1.
            # A partition is defined by an index i where 0 <= i < n - 1.
            # The possible values for i are 0, 1, ..., n-2.
            # There are (n-2) - 0 + 1 = n-1 such distinct indices.
            # Each such index defines a valid partition with non-empty subarrays
            # because the problem constraints state 2 <= n, so n-1 >= 1.
            # All these n-1 partitions will satisfy the condition.
            return n - 1