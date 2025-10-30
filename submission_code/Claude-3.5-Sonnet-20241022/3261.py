class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        def can_achieve_target(target: int) -> bool:
            # Try to eliminate bits larger than target
            mask = (1 << 30) - 1
            operations = 0
            i = 0
            
            while i < len(nums):
                # Keep track of current AND result
                curr = nums[i]
                
                # Try to combine with next numbers until we get result <= target
                while i < len(nums) - 1 and (curr & mask) > target:
                    i += 1
                    curr &= nums[i]
                    operations += 1
                
                i += 1
            
            return operations <= k
        
        # Binary search on answer
        left, right = 0, (1 << 30) - 1
        
        while left < right:
            mid = (left + right) >> 1
            if can_achieve_target(mid):
                right = mid
            else:
                left = mid + 1
                
        return left