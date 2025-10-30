class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        operations = 0
        nums.sort()
        for num in nums:
            if num < k:
                operations += 1
            else:
                break
        return operations