class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_sum = 0
        double_digit_sum = 0
        
        for num in nums:
            if num < 10:  # single-digit
                single_digit_sum += num
            else:  # double-digit
                double_digit_sum += num
        
        total_sum = single_digit_sum + double_digit_sum
        
        # Alice wins if she can choose a group where her sum > Bob's sum
        # If Alice chooses single-digit: Alice = single_digit_sum, Bob = double_digit_sum
        # If Alice chooses double-digit: Alice = double_digit_sum, Bob = single_digit_sum
        
        return single_digit_sum > double_digit_sum or double_digit_sum > single_digit_sum