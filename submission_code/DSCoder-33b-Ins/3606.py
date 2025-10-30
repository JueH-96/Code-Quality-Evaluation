class Solution:
    def minElement(self, nums: List[int]) -> int:
        def sum_of_digits(n):
            return sum(int(digit) for digit in str(n))

        nums = [sum_of_digits(n) for n in nums]
        return min(nums)