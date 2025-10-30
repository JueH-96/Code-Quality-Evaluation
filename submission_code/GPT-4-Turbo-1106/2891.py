class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        nums.sort()
        max_beauty = 1
        left = 0
        
        for right in range(1, len(nums)):
            # Ensure the difference between the current and the start of the window is within k
            while nums[right] - nums[left] > 2 * k:
                left += 1
            max_beauty = max(max_beauty, right - left + 1)
        
        return max_beauty