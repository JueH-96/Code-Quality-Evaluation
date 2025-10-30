from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        k_or_result = 0
        
        # We need to consider bits from position 0 up to 30.
        # The constraint nums[i] < 2^31 means that nums[i] can be at most 2^31 - 1.
        # The number 2^31 - 1 has bits 0 through 30 set (as 1s), and all higher bits are 0.
        # For example, if nums[i] could be up to 2^3 - 1 = 7 (binary 111),
        # we would check bits 0, 1, 2. This corresponds to range(3).
        # So for numbers up to 2^31 - 1, we check bits 0 to 30, which is range(31).
        for i in range(31):  # This loop iterates i from 0 to 30.
            
            # For the current bit position 'i', count how many numbers in 'nums'
            # have this bit set.
            bit_set_count = 0
            for num in nums:
                # To check if the i-th bit is set in 'num':
                # Right shift 'num' by 'i' positions. This moves the i-th bit to position 0.
                # Then, AND with 1. If the i-th bit of 'num' was 1, the result is 1. Otherwise, it's 0.
                if (num >> i) & 1:
                    bit_set_count += 1
            
            # According to the problem definition:
            # "The i^th bit is set in the K-or if and only if there are at least k elements
            # of nums in which bit i is set."
            if bit_set_count >= k:
                # If the condition is met, set the i-th bit in our result.
                # (1 << i) creates an integer that has only the i-th bit set (its value is 2^i).
                # Using bitwise OR (|) with 'k_or_result' sets this bit in k_or_result 
                # without affecting other bits already set in k_or_result.
                k_or_result = k_or_result | (1 << i)
                
        return k_or_result