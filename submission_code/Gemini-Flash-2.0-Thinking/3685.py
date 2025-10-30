class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n - 2):
            first = nums[i]
            second = nums[i+1]
            third = nums[i+2]
            if 2 * (first + third) == second:
                count += 1
        return count