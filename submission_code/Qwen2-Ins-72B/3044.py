class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        nums = nums[::-1]
        collected = set()
        operations = 0
        for num in nums:
            operations += 1
            if num <= k:
                collected.add(num)
            if len(collected) == k:
                return operations