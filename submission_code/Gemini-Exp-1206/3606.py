class Solution:
    def minElement(self, nums: List[int]) -> int:
        def sum_digits(n):
            s = 0
            while n:
                s += n % 10
                n //= 10
            return s

        min_val = float('inf')
        for num in nums:
            min_val = min(min_val, sum_digits(num))
        return min_val