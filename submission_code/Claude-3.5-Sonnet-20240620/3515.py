class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_sum = sum(num for num in nums if num < 10)
        double_digit_sum = sum(num for num in nums if num >= 10)
        total_sum = single_digit_sum + double_digit_sum
        
        return max(single_digit_sum, double_digit_sum) > total_sum // 2