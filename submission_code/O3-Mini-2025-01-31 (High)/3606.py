from typing import List

class Solution:
    def minElement(self, nums: List[int]) -> int:
        # Helper function to compute the sum of digits of a number
        def sum_of_digits(num: int) -> int:
            total = 0
            while num:
                total += num % 10
                num //= 10
            return total
        
        # Replace each element with its digit sum
        replaced = [sum_of_digits(num) for num in nums]
        # Return the minimum element among the replaced values
        return min(replaced)