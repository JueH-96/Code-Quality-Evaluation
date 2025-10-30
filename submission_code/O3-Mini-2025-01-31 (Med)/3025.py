from typing import List

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # First, if the total sum of nums is less than target, it's impossible.
        if sum(nums) < target:
            return -1
        
        # We will count the frequency of each power of two.
        # Given the constraints, the numbers are powers of 2 with nums[i] <= 2^30 and target < 2^31,
        # so we need to consider bit positions 0 to 31 (we add one extra slot for safety).
        MAX_BIT = 32
        freq = [0] * (MAX_BIT + 1)
        
        for num in nums:
            # find bit position: since num is a power of 2, one can use bit_length-1.
            bit = num.bit_length() - 1
            freq[bit] += 1
        
        operations = 0
        carry = 0  # Represents available pieces of current bit summed from lower splits.
        
        # Process each bit from 0 to MAX_BIT.
        for i in range(MAX_BIT):
            # At bit i, add pieces that come directly from our frequency.
            carry += freq[i]
            
            # Check if the i-th bit of target is 1.
            if (target >> i) & 1:
                # We need one piece of 2^i.
                if carry == 0:
                    # Borrow from a higher bit.
                    j = i + 1
                    while j <= MAX_BIT and freq[j] == 0:
                        j += 1
                    # If no higher bit is available, return -1.
                    if j > MAX_BIT:
                        return -1
                    # Now, split a piece from bit j down to bit i.
                    while j > i:
                        # Use one piece from bit j.
                        freq[j] -= 1
                        # Splitting a piece of 2^j produces 2 pieces of 2^(j-1).
                        freq[j - 1] += 2
                        operations += 1
                        j -= 1
                    # Now we have increased the count at bit i.
                    carry += 1  # One new piece from splitting is now available.
                # Use one piece for the current target bit.
                carry -= 1
            # Any leftover pieces at this bit can be paired to form pieces of the next bit.
            carry //= 2
        
        return operations