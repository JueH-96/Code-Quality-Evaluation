from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        # Convert numbers to their string representation once.
        # They all have the same length.
        n = len(nums)
        str_nums = [str(num) for num in nums]
        m = len(str_nums[0])
        total_diff = 0

        # For each digit position
        for pos in range(m):
            # Count frequency of each digit (0-9)
            freq = [0] * 10
            for s in str_nums:
                digit = int(s[pos])
                freq[digit] += 1

            # Number of unordered pairs that share the same digit at pos
            same_pairs = 0
            for count in freq:
                if count > 1:
                    same_pairs += count * (count - 1) // 2
            
            # Total pairs
            total_pairs = n * (n - 1) // 2
            diff_pairs = total_pairs - same_pairs
            total_diff += diff_pairs

        return total_diff