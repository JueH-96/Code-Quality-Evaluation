from typing import List

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        # Create a list of tuples where each tuple contains the number of set bits and the number itself
        nums_with_set_bits = [(bin(num).count('1'), num) for num in nums]
        
        # Sort the list of tuples based on the number of set bits and the number itself
        sorted_nums_with_set_bits = sorted(nums_with_set_bits)
        
        # Check if the sorted list of numbers is the same as the sorted list of original numbers
        return [num for _, num in sorted_nums_with_set_bits] == sorted(nums)