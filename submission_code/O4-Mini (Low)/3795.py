from typing import List

class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        # If nums already all zero, no queries needed.
        if all(x == 0 for x in nums):
            return 0
        
        n = len(nums)
        max_num = max(nums)
        # dp[i] is a bitmask where bit s is 1 if sum s can be formed for index i.
        dp = [(1 << 0) for _ in range(n)]  # initially only sum 0 is possible
        
        # We'll mask off bits above max_num to keep dp small.
        full_mask = (1 << (max_num + 1)) - 1
        
        # Process queries one by one.
        for k, (l, r, val) in enumerate(queries, start=1):
            shift = val
            for i in range(l, r + 1):
                # add new sums by shifting existing ones by 'val'
                dp[i] |= (dp[i] << shift) & full_mask
            
            # After applying this query, check if every nums[i] is achievable.
            all_zeroable = True
            for i, target in enumerate(nums):
                # if bit 'target' is not set, cannot zero this index yet
                if ((dp[i] >> target) & 1) == 0:
                    all_zeroable = False
                    break
            
            if all_zeroable:
                return k
        
        # If we finish all queries and still not all zeroable:
        return -1