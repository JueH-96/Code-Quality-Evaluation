class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        collected = set()
        operations = 0
        # Start from the end of nums and move backward
        for i in range(len(nums) - 1, -1, -1):
            operations += 1
            # If current element is in [1..k], add it to the collected set
            if 1 <= nums[i] <= k:
                collected.add(nums[i])
            # If we've collected all elements from 1..k, break
            if len(collected) == k:
                break
        return operations