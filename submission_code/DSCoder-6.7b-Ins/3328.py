class Solution:
    def minOperations(self, k: int) -> int:
        nums = [1]
        operations = 0
        while sum(nums) < k:
            max_val = max(nums)
            max_idx = nums.index(max_val)
            nums[max_idx] += 1
            operations += 1
            if sum(nums) < k:
                nums.append(max_val)
                operations += 1
        return operations