class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # If k is not in nums, it's impossible to make all elements equal to k
        if k not in nums:
            return -1

        # Sort the nums array in descending order
        nums.sort(reverse=True)

        # Initialize the operation count
        operations = 0

        # Iterate through the sorted nums array
        for num in nums:
            # If the current number is greater than k, decrement it to k
            if num > k:
                operations += 1
            # If the current number is less than or equal to k, stop the loop
            else:
                break

        return operations