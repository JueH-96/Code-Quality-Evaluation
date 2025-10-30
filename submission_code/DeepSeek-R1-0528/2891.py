class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        nums.sort()
        left = 0
        max_len = 0
        n = len(nums)
        for right in range(n):
            while nums[right] - nums[left] > 2 * k:
                left += 1
            max_len = max(max_len, right - left + 1)
        return max_len