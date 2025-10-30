from typing import List

class Solution:
    def minElement(self, nums: List[int]) -> int:
        def sum_digits(n):
            s = 0
            while n > 0:
                s += n % 10
                n = n // 10
            return s
        
        transformed = [sum_digits(num) for num in nums]
        return min(transformed)