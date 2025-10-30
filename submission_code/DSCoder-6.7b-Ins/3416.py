class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        # Convert the first number to string
        first_num_str = str(nums[0])
        # Initialize the sum of digit differences
        sum_diff = 0
        # Iterate over all the numbers
        for num in nums:
            # Convert the current number to string
            num_str = str(num)
            # Iterate over all the digits in the number
            for i in range(len(num_str)):
                # Calculate the digit difference between the first number and the current number
                diff = abs(int(first_num_str[i]) - int(num_str[i]))
                # Add the digit difference to the sum
                sum_diff += diff
        # Return the sum of digit differences
        return sum_diff