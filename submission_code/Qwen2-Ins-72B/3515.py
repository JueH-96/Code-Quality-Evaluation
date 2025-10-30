class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit_sum = sum(num for num in nums if num < 10)
        double_digit_sum = sum(num for num in nums if num >= 10)
        
        # Alice wins if the sum of single-digit numbers is strictly greater than the sum of double-digit numbers
        if single_digit_sum > double_digit_sum:
            return True
        
        # Alice wins if the sum of double-digit numbers is strictly greater than the sum of single-digit numbers
        if double_digit_sum > single_digit_sum:
            return True
        
        # Alice cannot win if the sums are equal
        return False