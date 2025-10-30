from typing import List

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # there are at most 31 relevant powers ( 2^0 … 2^30 )
        MAX_BIT = 31
        
        # count how many times every power of two occurs in nums
        cnt = [0] * MAX_BIT
        for x in nums:
            cnt[x.bit_length() - 1] += 1      # because x is a power of two
        
        # if the whole array is not big enough, it is impossible
        if sum(nums) < target:
            return -1
        
        operations = 0        # answer we have to return
        carry      = 0        # number of coins (value 2^i) we still have
                              # while processing bit i
        
        for i in range(MAX_BIT):
            carry += cnt[i]                   # coins already available at 2^i
            
            if (target >> i) & 1:             # we need one 2^i coin
                if carry == 0:                # none available – we have to split
                    j = i + 1
                    # look for the next larger coin that we can break
                    while j < MAX_BIT and cnt[j] == 0:
                        j += 1
                    if j == MAX_BIT:          # should not happen because total sum is large enough
                        return -1
                    # split the found coin step by step down to size 2^i
                    while j > i:
                        cnt[j]  -= 1          # take one coin of 2^j
                        cnt[j-1] += 2         # obtain two coins of 2^(j-1)
                        operations += 1       # one operation per split
                        j -= 1
                    carry += cnt[i]           # coins of size 2^i were created
                # use one 2^i coin for the target
                carry -= 1
            
            # every pair of remaining 2^i coins can act as one 2^(i+1) coin
            carry //= 2
        
        return operations