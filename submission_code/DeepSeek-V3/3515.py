class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_digit = [num for num in nums if num < 10]
        double_digit = [num for num in nums if num >= 10]
        
        sum_single = sum(single_digit)
        sum_double = sum(double_digit)
        
        total_sum = sum(nums)
        
        # If Alice chooses single-digit numbers
        alice_sum1 = sum_single
        bob_sum1 = total_sum - alice_sum1
        if alice_sum1 > bob_sum1:
            return True
        
        # If Alice chooses double-digit numbers
        alice_sum2 = sum_double
        bob_sum2 = total_sum - alice_sum2
        if alice_sum2 > bob_sum2:
            return True
        
        return False