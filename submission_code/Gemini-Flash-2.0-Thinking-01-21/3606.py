class Solution:
    def minElement(self, nums: List[int]) -> int:
        def sum_digits(num):
            s = 0
            for digit in str(num):
                s += int(digit)
            return s
        
        digit_sums = []
        for num in nums:
            digit_sums.append(sum_digits(num))
        
        return min(digit_sums)