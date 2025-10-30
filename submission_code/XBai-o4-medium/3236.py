class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        current_length = 1
        n = len(nums)
        for i in range(1, n):
            if nums[i] == nums[i-1] + 1:
                current_length += 1
            else:
                break
        sum_val = sum(nums[:current_length])
        num_set = set(nums)
        while True:
            if sum_val not in num_set:
                return sum_val
            sum_val += 1