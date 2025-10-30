class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        required = set(range(1, k + 1))
        collected = set()
        ops = 0
        for num in reversed(nums):
            collected.add(num)
            ops += 1
            if required.issubset(collected):
                return ops