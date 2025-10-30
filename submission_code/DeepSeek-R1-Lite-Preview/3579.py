from functools import cmp_to_key
from typing import List

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        # Function to compare two binary strings for sorting
        def compare(a, b):
            if a + b > b + a:
                return -1  # a should come before b
            elif a + b < b + a:
                return 1   # b should come before a
            else:
                return 0
        
        # Convert each number to its binary representation without '0b'
        binary_list = [format(num, 'b') for num in nums]
        
        # Sort the binary strings using the custom comparator
        binary_list.sort(key=cmp_to_key(compare))
        
        # Concatenate the sorted binary strings
        concatenated_binary = ''.join(binary_list)
        
        # Convert the concatenated binary string to an integer
        result = int(concatenated_binary, 2)
        
        return result