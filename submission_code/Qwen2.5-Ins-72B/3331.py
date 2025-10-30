class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        operations = 0
        for num in sorted(nums):
            if num < k:
                operations += 1
            else:
                break
        return operations