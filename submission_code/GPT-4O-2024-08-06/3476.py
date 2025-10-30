class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        # Initialize counters for numbers with remainders 1 and 2 when divided by 3
        remainder_1_count = 0
        remainder_2_count = 0
        
        # Count the numbers with remainder 1 and 2
        for num in nums:
            if num % 3 == 1:
                remainder_1_count += 1
            elif num % 3 == 2:
                remainder_2_count += 1
        
        # The minimum operations needed is to change all remainder 1 to 0 or all remainder 2 to 0
        # Each change requires 1 operation, so we take the sum of both counts
        return remainder_1_count + remainder_2_count