from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        # Number of numbers
        n = len(nums)
        if n < 2:
            return 0
        
        # Determine number of digits (all numbers have same length)
        # Convert first number to string to get digit count
        d = len(str(nums[0]))
        
        # Precompute divisors to extract each digit position:
        # For position p = 0..d-1, divisor = 10^(d-1-p)
        divisors = [10 ** (d - 1 - p) for p in range(d)]
        
        total = 0
        # For each digit position, count frequency of each digit 0..9
        for div in divisors:
            cnt = [0] * 10
            for num in nums:
                digit = (num // div) % 10
                cnt[digit] += 1
            
            # Number of ordered pairs at this position with differing digits:
            # sum_{digit} cnt[d] * (n - cnt[d])
            ordered_diff = sum(c * (n - c) for c in cnt)
            # Each unordered pair is counted twice in ordered_diff,
            # so divide by 2 to get the count of differing pairs.
            total += ordered_diff // 2
        
        return total