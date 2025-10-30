class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0

        while n > 0:
            # Check if the current window of 3 elements has distinct elements
            if n >= 3 and len(set(nums[:3])) == 3:
                break
            # Remove the first 3 elements
            nums = nums[3:]
            n -= 3
            operations += 1

        return operations