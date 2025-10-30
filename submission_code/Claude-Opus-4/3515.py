class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_sum = 0
        double_digit_sum = 0
        
        for num in nums:
            if num < 10:
                single_digit_sum += num
            else:
                double_digit_sum += num
        
        # If Alice chooses single-digit numbers, Bob gets double-digit numbers
        # Alice wins if single_digit_sum > double_digit_sum
        
        # If Alice chooses double-digit numbers, Bob gets single-digit numbers
        # Alice wins if double_digit_sum > single_digit_sum
        
        return single_digit_sum > double_digit_sum or double_digit_sum > single_digit_sum