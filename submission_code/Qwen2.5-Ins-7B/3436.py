class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        current_or = 0
        
        for i in range(len(nums)):
            current_or |= nums[i]
            diff = abs(current_or - k)
            min_diff = min(min_diff, diff)
            
            if current_or >= k:
                break
        
        return min_diff