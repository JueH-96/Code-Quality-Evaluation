class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        # Sort the array to prioritize the largest elements for operations
        nums.sort(reverse=True)

        # Apply Operation 2 first to maximize the reduction
        for i in range(len(nums)):
            if op2 > 0 and nums[i] >= k:
                nums[i] -= k
                op2 -= 1

        # Apply Operation 1 to the remaining elements
        for i in range(len(nums)):
            if op1 > 0:
                nums[i] = (nums[i] + 1) // 2
                op1 -= 1

        # Return the sum of the array after operations
        return sum(nums)