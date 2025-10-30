class Solution:
    def minOperations(self, k: int) -> int:
        nums = [1]
        ops = 0
        while sum(nums) < k:
            if nums[-1] * 2 < k:
                nums.append(nums[-1] * 2)
            else:
                nums.append(nums[-1] + 1)
            ops += 1
        return ops