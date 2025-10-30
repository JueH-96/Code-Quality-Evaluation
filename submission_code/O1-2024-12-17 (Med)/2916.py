class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        # If the array has only one element, we don't need to split anything
        if n == 1:
            return True
        
        # Precompute prefix sums for fast range-sum queries
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        # Helper to get sum of subarray nums[l:r+1]
        def sub_sum(l, r):
            return prefix_sum[r + 1] - prefix_sum[l]
        
        from functools import lru_cache
        
        @lru_cache(None)
        def can_split(l, r):
            # If the subarray has length 1, it's already a valid single-element array
            if r - l + 1 == 1:
                return True
            # Try all possible splits between l and r
            for mid in range(l, r):
                left_sum = sub_sum(l, mid)
                right_sum = sub_sum(mid + 1, r)
                left_len = mid - l + 1
                right_len = r - (mid + 1) + 1
                # Check splitting validity conditions
                if ((left_sum >= m or left_len == 1) and 
                    (right_sum >= m or right_len == 1) and 
                    can_split(l, mid) and 
                    can_split(mid + 1, r)):
                    return True
            return False
        
        # Check if we can split the entire array from index 0 to n-1
        return can_split(0, n - 1)