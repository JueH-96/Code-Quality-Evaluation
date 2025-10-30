from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return 0  # No pairs to compare
        
        # Convert nums to list of strings
        str_nums = [str(num) for num in nums]
        m = len(str_nums[0])  # All numbers have the same number of digits
        
        total_sum = 0
        # Precompute total pairs
        total_pairs = n * (n - 1) // 2
        
        for pos in range(m):
            freq = [0] * 10  # Frequency of digits 0-9 at this position
            for num_str in str_nums:
                digit = int(num_str[pos])
                freq[digit] += 1
            # Calculate sum of C(freq[d], 2) for d in 0-9
            same_pairs = sum(f * (f - 1) // 2 for f in freq)
            # Different pairs at this position
            diff_pairs = total_pairs - same_pairs
            # Accumulate to total sum
            total_sum += diff_pairs
        
        return total_sum