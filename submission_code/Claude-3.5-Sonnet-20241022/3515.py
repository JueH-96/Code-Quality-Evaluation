class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digits = []
        double_digits = []
        
        # Separate numbers into single and double digit lists
        for num in nums:
            if num < 10:
                single_digits.append(num)
            else:
                double_digits.append(num)
                
        # Calculate sums
        single_sum = sum(single_digits)
        double_sum = sum(double_digits)
        total_sum = single_sum + double_sum
        
        # If Alice takes single digits, Bob gets double digits
        if single_sum > double_sum:
            return True
            
        # If Alice takes double digits, Bob gets single digits
        if double_sum > single_sum:
            return True
            
        return False