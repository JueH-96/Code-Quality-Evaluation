from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        # diff array to track flip parity ending at each index
        diff = [0] * (n + 3)
        curr_flip = 0
        ops = 0
        
        for i in range(n):
            # apply any flip that ends at position i
            curr_flip ^= diff[i]
            # if after current flips, nums[i] is 0, we must flip at i
            if nums[i] ^ curr_flip == 0:
                # can't flip if we don't have 3 elements from i
                if i + 2 >= n:
                    return -1
                ops += 1
                # start a flip at i (toggle curr_flip)
                curr_flip ^= 1
                # that flip will end after i+2, so we cancel it at i+3
                diff[i + 3] ^= 1
        
        return ops