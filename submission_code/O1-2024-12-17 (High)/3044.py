class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        collected = set()
        operations = 0
        i = len(nums) - 1
        
        # Remove from the end until we've collected all numbers in [1..k].
        while len(collected) < k:
            if 1 <= nums[i] <= k:
                collected.add(nums[i])
            operations += 1
            i -= 1
        
        return operations