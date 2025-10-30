class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        max_value = 0
        min_i = nums[0]

        for j in range(1, len(nums) - 1):
            min_i = min(min_i, nums[j - 1])
            for k in range(j + 1, len(nums)):
                max_value = max(max_value, (nums[j] - min_i) * nums[k])

        return max_value