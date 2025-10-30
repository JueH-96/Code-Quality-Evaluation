from typing import List

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Quick check: if total sum is less than target, impossible
        if sum(nums) < target:
            return -1
        
        # Count how many powers of two we have for each bit position
        MAXB = 31  # handle bits 0..30 (target < 2^31)
        cnt = [0] * (MAXB + 1)
        for x in nums:
            b = x.bit_length() - 1
            cnt[b] += 1
        
        ops = 0
        carry = 0  # how many 2^i units we have available at current level
        
        # Process bits from 0 up to MAXB
        for i in range(MAXB + 1):
            # Add all available 2^i pieces to carry
            carry += cnt[i]
            
            # If target has the i-th bit set, we need one 2^i
            if (target >> i) & 1:
                if carry > 0:
                    # use one
                    carry -= 1
                else:
                    # need to split a higher bit
                    j = i + 1
                    while j <= MAXB and cnt[j] == 0:
                        j += 1
                    if j > MAXB:
                        return -1
                    # split down from bit j to bit i
                    # each split of 2^k -> two 2^(k-1) costs one operation
                    while j > i:
                        cnt[j] -= 1
                        cnt[j - 1] += 2
                        ops += 1
                        j -= 1
                    # now we have cnt[i] >= 2 from the splits
                    carry += cnt[i]
                    carry -= 1  # use one for our need
                # after using one, carry is updated
            # any leftover carry of 2^i can form carry//2 of 2^(i+1)
            carry //= 2
        
        return ops