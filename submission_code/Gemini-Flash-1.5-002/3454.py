class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        diff = [target[i] - nums[i] for i in range(len(nums))]
        count = 0
        curr_sum = 0
        for i in range(len(diff)):
            curr_sum += diff[i]
            if curr_sum != 0:
                count += 1
                curr_sum = 0
        return count