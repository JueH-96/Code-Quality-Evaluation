class Solution:
    def minElement(self, nums: List[int]) -> int:
        def sum_of_digits(n):
            return sum(int(digit) for digit in str(n))
        
        replaced_nums = [sum_of_digits(num) for num in nums]
        return min(replaced_nums)