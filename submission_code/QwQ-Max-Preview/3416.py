from collections import Counter
from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        # Precompute the digits for each number
        digits_list = []
        d = len(str(nums[0]))  # Determine the number of digits per number
        for num in nums:
            s = str(num)
            digits = list(map(int, s))
            digits_list.append(digits)
        
        total = 0
        n = len(nums)
        for i in range(d):
            # Collect digits in the current position for all numbers
            current_digits = [digits[i] for digits in digits_list]
            # Count the frequency of each digit
            cnt = Counter(current_digits)
            # Calculate total possible pairs
            total_pairs = n * (n - 1) // 2
            # Calculate pairs with the same digit
            same_pairs = sum(c * (c - 1) // 2 for c in cnt.values())
            # Add the differing pairs for this position to the total
            total += (total_pairs - same_pairs)
        
        return total