class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # A bitwise OR has at least one trailing zero (i.e., is even)
        # only if all selected numbers are even. Thus we simply
        # check if there are at least two even numbers in the array.
        even_count = sum(1 for x in nums if x % 2 == 0)
        return even_count >= 2