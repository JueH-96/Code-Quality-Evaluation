class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # Iterate through each number in the list
        for i in range(len(nums)):
            # Check the bitwise OR of the current number with each subsequent number
            for j in range(i + 1, len(nums)):
                # Calculate the bitwise OR of the two numbers
                bitwise_or = nums[i] | nums[j]
                # Check if the bitwise OR has at least one trailing zero
                if bitwise_or & 1 == 0:
                    return True
        # If no such pair is found, return False
        return False