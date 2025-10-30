class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        sum_single_digit = sum(num for num in nums if 1 <= num <= 9)
        sum_double_digit = sum(num for num in nums if 10 <= num <= 99)
        total_sum = sum(nums)
        
        # If Alice chooses single-digit numbers
        if sum_single_digit > total_sum - sum_single_digit:
            return True
        
        # If Alice chooses double-digit numbers
        if sum_double_digit > total_sum - sum_double_digit:
            return True
        
        return False