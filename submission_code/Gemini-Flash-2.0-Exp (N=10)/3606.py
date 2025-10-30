class Solution:
    def minElement(self, nums: List[int]) -> int:
        def sum_digits(n):
            s = 0
            while n:
                s += n % 10
                n //= 10
            return s

        transformed_nums = [sum_digits(num) for num in nums]
        return min(transformed_nums)