class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        nums.sort(reverse=True)
        total = 0
        for i in range(len(nums)):
            if i % 3 == 2:
                total += nums[i]
        return total