class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        target = set(range(1, k + 1))
        collected = set()
        operations = 0
        
        for num in reversed(nums):
            operations += 1
            if num in target:
                collected.add(num)
                if collected == target:
                    return operations
        
        return operations