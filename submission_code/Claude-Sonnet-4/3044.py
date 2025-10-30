class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Set to keep track of numbers from 1 to k that we've collected
        collected = set()
        operations = 0
        
        # Work backwards from the end of the array
        for i in range(len(nums) - 1, -1, -1):
            operations += 1
            
            # If the current number is in range [1, k], add it to collected
            if 1 <= nums[i] <= k:
                collected.add(nums[i])
            
            # If we've collected all numbers from 1 to k, we're done
            if len(collected) == k:
                break
        
        return operations