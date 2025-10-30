class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # We need at least two elements in the selection whose bitwise OR ends with zero.
        # A binary number ends with zero (i.e., has at least one trailing zero) if and only if it is even.
        # The OR of selected numbers is even if and only if all selected numbers are even.
        # Therefore, we just need to check if there are at least two even numbers in nums.
        
        even_count = sum(1 for x in nums if x % 2 == 0)
        return even_count >= 2