from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Determine the maximal bit index we need to consider.
        # We must consider all bits that might be 1 either in k or any element in nums.
        max_val = k
        for num in nums:
            if num > max_val:
                max_val = num
        # If max_val is 0 then we only need 1 bit.
        bits = max_val.bit_length() if max_val > 0 else 1

        res = 0
        # for each bit position b, compute the parity (0 or 1) of ones in that position for nums.
        for b in range(bits):
            ones = 0
            mask = 1 << b
            # Count ones in bit position b.
            for num in nums:
                if num & mask:
                    ones += 1
            parity = ones & 1  # parity of this bit position in the XOR of nums (without any changes)

            # k's b-th bit:
            kbit = 1 if (k & mask) else 0

            # We want: (parity + (total flips on bit b mod2)) mod2 = kbit.
            # Let f = (total flips on bit b mod2). Then we require:
            #   parity ⊕ f = kbit   so   f = parity ⊕ kbit.
            # f must be 0 or 1.
            # We can always achieve cost f by flipping the respective bit in exactly f element(s)
            # (if f==1, choose any one element and flip the b-th bit; if f==0, do nothing).
            # So the minimal cost for bit b is exactly (parity ⊕ kbit).
            res += (parity ^ kbit)
        return res