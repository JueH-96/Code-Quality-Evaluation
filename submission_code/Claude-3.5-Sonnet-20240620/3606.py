class Solution:
    def minElement(self, nums: List[int]) -> int:
        def sum_digits(num):
            return sum(int(digit) for digit in str(num))
        
        return min(sum_digits(num) for num in nums)