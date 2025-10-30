class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        nums.sort()
        left, right = 0, 0
        max_beauty = 0
        while right < len(nums):
            if nums[right] - nums[left] > 2 * k:
                left += 1
            else:
                max_beauty = max(max_beauty, right - left + 1)
                right += 1
        return max_beauty