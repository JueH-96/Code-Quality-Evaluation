class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_sum = 0
        double_sum = 0
        
        for num in nums:
            if num < 10:  # single-digit (1-9)
                single_sum += num
            else:  # double-digit (10-99)
                double_sum += num
        
        # Alice can win if the sums are different
        # If single_sum > double_sum, Alice chooses single-digit
        # If double_sum > single_sum, Alice chooses double-digit
        # If they're equal, Alice can't win
        return single_sum != double_sum