class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        max_beauty = 0
        
        left = 0
        for right in range(n):
            # Expand window while the condition is violated
            while nums[right] - nums[left] > 2 * k:
                left += 1
            # Update max_beauty with current window size
            max_beauty = max(max_beauty, right - left + 1)
        
        return max_beauty