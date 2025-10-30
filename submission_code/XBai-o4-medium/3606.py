class Solution:
    def minElement(self, nums: List[int]) -> int:
        min_val = float('inf')
        for num in nums:
            digit_sum = sum(int(digit) for digit in str(num))
            if digit_sum < min_val:
                min_val = digit_sum
        return min_val