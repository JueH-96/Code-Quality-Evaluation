class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        nums.sort()
        left = 0
        max_beauty = 0
        
        for right in range(len(nums)):
            # Shrink window while the difference is too large
            while nums[right] - nums[left] > 2 * k:
                left += 1
            
            # Update maximum beauty
            max_beauty = max(max_beauty, right - left + 1)
        
        return max_beauty