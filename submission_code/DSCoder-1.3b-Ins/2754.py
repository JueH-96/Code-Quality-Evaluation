class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        # Initialize the result to 1 because any number multiplied by 1 is the number itself
        result = 1

        # Initialize the sign to 1 because we want to maximize the strength
        sign = 1

        # Iterate over the list of numbers
        for num in nums:
            # If the number is negative, change the sign and make it positive
            if num < 0:
                sign *= -1
                num = -num

            # Multiply the result by the current number
            result *= num

        # Return the result after all numbers have been processed
        return result