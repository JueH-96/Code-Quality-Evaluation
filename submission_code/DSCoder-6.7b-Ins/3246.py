class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # Initialize a variable to store the bitwise OR of all numbers
        or_result = 0
        # Iterate over all numbers in the input list
        for num in nums:
            # Bitwise OR the current number with the result
            or_result |= num
        # Convert the result to binary and store it as a string
        bin_result = bin(or_result)[2:]
        # Check if the binary string ends with a '0'
        return '0' in bin_result[-1]