class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_sum = -1
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if max(int(nums[i]), int(nums[j])) == int(nums[i]) and int(nums[i]) == int(nums[j]):
                    max_sum = max(max_sum, int(nums[i]) + int(nums[j]))
        return max_sum