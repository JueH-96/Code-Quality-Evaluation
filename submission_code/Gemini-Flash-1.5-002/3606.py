class Solution:
    def minElement(self, nums: List[int]) -> int:
        min_num = float('inf')
        for num in nums:
            sum_digits = sum(int(digit) for digit in str(num))
            min_num = min(min_num, sum_digits)
        return min_num