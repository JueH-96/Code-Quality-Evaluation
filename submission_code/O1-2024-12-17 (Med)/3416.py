from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        # Find the number of digits (D) in each integer (they all have the same length)
        D = len(str(nums[0]))
        n = len(nums)
        
        # digit_counts[d][x] will store how many numbers have digit x in position d
        # We'll index positions from right to left (0-based for the least significant digit)
        digit_counts = [[0]*10 for _ in range(D)]
        
        # Fill the digit_counts table
        for num in nums:
            x = num
            for d in range(D):
                digit_counts[d][x % 10] += 1
                x //= 10
        
        # Calculate the sum of digit differences
        # For each digit position, the total number of pairs is n*(n-1)//2
        # For each position, we subtract from that the number of pairs that have the same digit
        # which is sum( counts[d][digit]*(counts[d][digit]-1)//2 for digit in 0..9 )
        total_diff = 0
        total_pairs = n*(n-1)//2
        
        for d in range(D):
            same_pairs = 0
            for digit in range(10):
                count = digit_counts[d][digit]
                same_pairs += count*(count-1)//2
            total_diff += (total_pairs - same_pairs)
        
        return total_diff