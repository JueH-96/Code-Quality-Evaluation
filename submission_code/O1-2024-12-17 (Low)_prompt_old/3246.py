class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # A bitwise OR has at least one trailing zero (i.e., is even)
        # if and only if the result's least significant bit is 0.
        #
        # For the OR of two or more numbers to end with a 0 bit:
        # - Either all selected numbers are even (thus, least significant bit is 0).
        # - If we mix an even number with any odd number, the least significant bit becomes 1 (making it odd).
        #
        # Thus, a necessary and sufficient condition is having at least two even numbers.
        
        even_count = sum(1 for num in nums if num % 2 == 0)
        return even_count >= 2