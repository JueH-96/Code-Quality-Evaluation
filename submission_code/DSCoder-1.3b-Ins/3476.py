class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        nums = [n for n in nums if n != 0]
        return len(nums)