class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        collected = set()
        operations = 0
        
        # Traverse from the end until we collect all numbers from 1..k
        for i in range(len(nums) - 1, -1, -1):
            operations += 1
            if nums[i] <= k and nums[i] not in collected:
                collected.add(nums[i])
                if len(collected) == k:
                    return operations
        
        return operations