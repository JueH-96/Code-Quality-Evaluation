from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        # Convert every number to string to easily access digits by position.
        # All numbers have the same number of digits.
        num_strs = list(map(str, nums))
        length = len(num_strs[0])
        n = len(nums)
        total_sum = 0
        
        # For each digit position, count frequency for each digit.
        for pos in range(length):
            count = [0] * 10  # digits 0-9
            for s in num_strs:
                # Each s is already a string with equal length.
                digit = int(s[pos])
                count[digit] += 1
                
            # Total pairs for this position.
            total_pairs = n * (n - 1) // 2
            # Count pairs that have the same digit in this position.
            same_pairs = 0
            for freq in count:
                if freq > 1:
                    same_pairs += freq * (freq - 1) // 2
            # Pairs with different digits is total pairs minus same digit pairs.
            total_sum += total_pairs - same_pairs
        
        return total_sum