from typing import List

class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        """
        For every index we keep a bit-set of all sums that can be formed with the
        values (val) of the queries that touched this index so far.
        A bit at position s is 1  <=>  we can obtain a total decrement of s.
        
        After processing a prefix of length k, nums can be turned into all zeros
        iff  for every index i   the bit corresponding to nums[i] is set.
        """
        n = len(nums)
        
        # Already all zeros?
        if all(v == 0 for v in nums):
            return 0
        
        max_num = max(nums)                 # largest target we need to reach
        mask    = (1 << (max_num + 1)) - 1  # keep bit-sets shortened to max_num
        
        bits = [1] * n          # only sum 0 reachable initially  (bit 0 == 1)
        
        for k, (l, r, val) in enumerate(queries, 1):
            # update reachable sums for every index inside the range
            for idx in range(l, r + 1):
                bits[idx] |= (bits[idx] << val) & mask
            
            # check whether every index can already reach its required sum
            if all((bits[i] >> nums[i]) & 1 for i in range(n)):
                return k
        
        # even after all queries it is impossible
        return -1