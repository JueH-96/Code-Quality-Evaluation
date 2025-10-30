class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_sum = 0
        double_digit_sum = 0
        
        for num in nums:
            if 1 <= num <= 9:
                single_digit_sum += num
            elif 10 <= num <= 99:
                double_digit_sum += num
        
        total_sum = sum(nums)
        
        # Alice chooses single-digit numbers
        if single_digit_sum > total_sum - single_digit_sum:
            return True
        
        # Alice chooses double-digit numbers
        if double_digit_sum > total_sum - double_digit_sum:
            return True
        
        return False