class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        # segregate sums of single-digit and double-digit numbers  
        sum_single = sum(num for num in nums if num < 10)
        sum_double = sum(num for num in nums if num >= 10)
        total = sum(nums)
        
        # Option 1: Alice chooses all single-digit numbers
        bob_sum_when_single = total - sum_single
        alice_wins_single = sum_single > bob_sum_when_single
        
        # Option 2: Alice chooses all double-digit numbers
        bob_sum_when_double = total - sum_double
        alice_wins_double = sum_double > bob_sum_when_double
        
        return alice_wins_single or alice_wins_double