class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        collected = set()
        operations = 0
        for num in reversed(nums):
            operations += 1
            if num <= k and num not in collected:
                collected.add(num)
                if len(collected) == k:
                    return operations
        return operations  # This line is theoretically unreachable due to problem constraints.