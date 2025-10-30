class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        nums.sort()
        max_beauty = 0
        right = 0
        for left in range(len(nums)):
            while right < len(nums) and nums[right] - nums[left] <= 2 * k:
                right += 1
            max_beauty = max(max_beauty, right - left)
        return max_beauty