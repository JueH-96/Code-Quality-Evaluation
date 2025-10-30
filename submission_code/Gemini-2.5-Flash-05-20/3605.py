import math
from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for N in nums:
            # Step 1: If N is even, it's impossible to satisfy x OR (x+1) == N.
            # This is because x OR (x+1) always results in an odd number.
            # (Proof: if x is even, x|(x+1) = x+1, which is odd.
            #  If x is odd, x has form P01...1. x+1 has form P10...0.
            #  x|(x+1) has form P11...1, which is odd as its LSB is 1).
            # Thus, if N is even, no solution exists, so ans[i] = -1.
            if N % 2 == 0:
                ans.append(-1)
                continue

            # Step 2: N is odd. We need to find the smallest non-negative x.
            current_min_x = float('inf')

            # We iterate through possible values of 'k', where 'k' is a bit position
            # (from 0 up to the most significant bit of N).
            # The maximum value for nums[i] is 1000. 1000 in binary is 1111101000_2.
            # N.bit_length() gives the number of bits required to represent N.
            # For N <= 1000, N.bit_length() is at most 10 (bits 0-9).
            # So, 'k' can range from 0 up to N.bit_length() - 1.
            for k in range(N.bit_length()): 
                # `mask_ones` is a number with `k+1` consecutive 1s (from bit 0 to bit k).
                mask_ones = (1 << (k + 1)) - 1
                
                # Check if N has all bits from 0 to k set to 1.
                # If `(N & mask_ones) == mask_ones` is true, it means N has the form P11...1
                # where the block of 1s (from bit 0 to k) is `k+1` bits long.
                # This structure implies that N can be formed by `x OR (x+1)` where
                # `x` has its rightmost 0-bit at position `k`.
                if (N & mask_ones) == mask_ones:
                    # If N has this structure, a candidate 'x' can be derived by taking N
                    # and flipping its k-th bit from 1 to 0.
                    # This candidate_x will have its k-th bit as 0 and all bits 0 to k-1 as 1,
                    # effectively making it of the form P01...1 (where 0 is at bit k, and 1s are below).
                    candidate_x = N ^ (1 << k)
                    
                    # For such a `candidate_x` (of form P01...1), it is guaranteed that
                    # `candidate_x | (candidate_x + 1)` will equal `N`.
                    # (This is because `candidate_x + 1` would be P10...0, and their OR yields P11...1, which is N).
                    # We are looking for the smallest such `x`, so we update `current_min_x`.
                    current_min_x = min(current_min_x, candidate_x)
            
            # As explained, if N is odd, N-1 is always a valid candidate for x (corresponding to k=0).
            # This ensures `current_min_x` will always be updated from `float('inf')` to an integer.
            ans.append(current_min_x)

        return ans