class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        nums.sort()
        l, r = 0, len(nums) - 1
        while l < r:
            averages.append((nums[l] + nums[r]) / 2)
            l += 1
            r -= 1
        return min(averages)