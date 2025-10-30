class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        operations = 0
        while nums and nums[0] < k:
            nums.pop(0)
            operations += 1
        return operations