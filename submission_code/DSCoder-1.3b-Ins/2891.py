class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        nums.sort()
        l, r = 0, len(nums) - 1
        res = beauty = 0
        while l <= r:
            if nums[r] - nums[l] > k:
                beauty -= nums[l]
                l += 1
            else:
                beauty = max(beauty, nums[r] + r - l)
                r -= 1
        return beauty