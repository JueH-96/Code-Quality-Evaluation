from typing import List

class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        # Maximum absolute alternating sum that can be produced
        total_sum = sum(nums)
        if abs(k) > total_sum:        # k is outside the achievable range
            return -1

        offset = total_sum                    # shift sums so they are non–negative indices
        BITS = offset * 2 + 1                 # number of different sums
        MASK = (1 << BITS) - 1                # to truncate bitsets
        
        size = limit + 1                      # product index goes from 0 … limit
        
        # dp_even / dp_odd  : bitset of sums reachable (may include empty subsequence)
        # dp_even_ne / dp_odd_ne : bitset of sums reachable with NON-EMPTY subsequence
        dp_even     = [0] * size
        dp_odd      = [0] * size
        dp_even_ne  = [0] * size
        dp_odd_ne   = [0] * size
        
        # empty subsequence – even length, product 1, alternating sum 0
        dp_even[1] = 1 << offset
        
        for num in nums:
            new_even    = dp_even[:]     # copy – case “skip num”
            new_odd     = dp_odd[:]
            new_even_ne = dp_even_ne[:]  # copy already existing non-empty states
            new_odd_ne  = dp_odd_ne[:]
            
            for p in range(size):
                bits_even = dp_even[p]
                if bits_even:
                    np = 0 if num == 0 else p * num
                    if np <= limit:
                        shifted = (bits_even << num) & MASK     # add ‘num’  (even → odd)
                        if shifted:
                            new_odd[np]     |= shifted
                            new_odd_ne[np]  |= shifted          # result is non-empty
                            
                bits_odd = dp_odd[p]
                if bits_odd:
                    np = 0 if num == 0 else p * num
                    if np <= limit:
                        shifted = bits_odd >> num               # subtract ‘num’ (odd → even)
                        if shifted:
                            new_even[np]    |= shifted
                            new_even_ne[np] |= shifted           # result is non-empty
            
            # assign for next round
            dp_even, dp_odd     = new_even, new_odd
            dp_even_ne, dp_odd_ne = new_even_ne, new_odd_ne
        
        target_idx = k + offset          # bit position that represents alternating sum == k
        if target_idx < 0 or target_idx >= BITS:
            return -1                    # k still outside representable range
        
        target_bit = 1 << target_idx
        # scan products from largest to smallest
        for prod in range(limit, -1, -1):
            if (dp_even_ne[prod] & target_bit) or (dp_odd_ne[prod] & target_bit):
                return prod
        return -1