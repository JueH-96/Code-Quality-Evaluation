class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        if k == 0:
            return reduce(lambda a, b: a | b, nums)
        for _ in range(k):
            min_val = min(nums[i] & nums[i+1] for i in range(len(nums)-1))
            nums = [min_val] + [nums[i] for i in range(len(nums)) if nums[i] != min_val]
        return reduce(lambda a, b: a | b, nums)