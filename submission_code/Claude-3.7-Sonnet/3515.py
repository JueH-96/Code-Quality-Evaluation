class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        # Calculate sum of single-digit numbers (1-9)
        single_digit_sum = sum(num for num in nums if num < 10)
        
        # Calculate sum of double-digit numbers (10-99)
        double_digit_sum = sum(num for num in nums if num >= 10)
        
        # Alice wins if either choice gives her a strictly larger sum
        # If she takes single-digit numbers, Bob gets double-digit numbers
        if single_digit_sum > double_digit_sum:
            return True
            
        # If she takes double-digit numbers, Bob gets single-digit numbers
        if double_digit_sum > single_digit_sum:
            return True
            
        # If both sums are equal, Alice can't win
        return False