from typing import List

class Solution:
    def minElement(self, nums: List[int]) -> int:
        def sum_of_digits(n: int) -> int:
            return sum(int(digit) for digit in str(n))
        
        min_sum = float('inf')
        for num in nums:
            current_sum = sum_of_digits(num)
            if current_sum < min_sum:
                min_sum = current_sum
        return min_sum