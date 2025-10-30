class Solution:
    def maximumOR(self, nums: List[int], k: int) -> int:
        result = 0
        max_bit = 30  # since nums[i] can be up to 1e9, which is less than 2^30
        k_remaining = k
        
        for i in reversed(range(max_bit + 1)):
            min_m = float('inf')
            for x in nums:
                # Find the highest set bit in x that is <= i
                set_bits = []
                temp = x
                j = -1
                while temp > 0:
                    j = max(j, (temp & -temp).bit_length() - 1)
                    temp >>= 1
                # Now find the largest j <= i
                if j <= i:
                    m = i - j
                    if m < min_m:
                        min_m = m
            if min_m != float('inf') and min_m <= k_remaining:
                result |= (1 << i)
                k_remaining -= min_m
        return result