class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        '''
        Determines if the binary representation of the bitwise OR of any combination of two or more elements has at least one trailing zero.

        Args:
            nums (List[int]): A list of positive integers.

        Returns:
            bool: True if there exists at least one combination of two or more elements whose bitwise OR results in a number with at least one trailing zero in its binary representation, false otherwise.
        '''
        from functools import reduce
        if len(nums) < 2:
            return False
        
        # Function to check if a number contains at least one trailing zero in its binary representation.
        def has_trailing_zero(num):
            return num & 1 != num

        # Using reduce to apply the bitwise OR operation on all elements.
        bitwise_or = reduce(lambda x, y: x | y, nums)

        # Check if the result has trailing zeros.
        return has_trailing_zero(bitwise_or)