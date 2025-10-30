class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        from functools import lru_cache
        
        @lru_cache(None)
        def can_split(start, end):
            if start == end:
                return True
            
            current_sum = prefix_sums[end + 1] - prefix_sums[start]
            if current_sum < m:
                return False
            
            for mid in range(start, end):
                left_sum = prefix_sums[mid + 1] - prefix_sums[start]
                right_sum = prefix_sums[end + 1] - prefix_sums[mid + 1]
                
                if (left_sum >= m or mid + 1 == start + 1) and (right_sum >= m or end == mid + 1):
                    if can_split(start, mid) and can_split(mid + 1, end):
                        return True
            
            return False
        
        n = len(nums)
        prefix_sums = [0] * (n + 1)
        
        for i in range(n):
            prefix_sums[i + 1] = prefix_sums[i] + nums[i]
        
        return can_split(0, n - 1)