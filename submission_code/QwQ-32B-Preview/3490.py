from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Convert nums to parities: 0 for even, 1 for odd
        parities = [num % 2 for num in nums]
        
        # Count the number of parity 0 and parity 1
        count_0 = parities.count(0)
        count_1 = parities.count(1)
        
        # Maximum count of one parity
        max_same_parity = max(count_0, count_1)
        
        # Compute the longest alternating parity subsequence
        dp0 = 0  # Length of subsequence ending with parity 0
        dp1 = 0  # Length of subsequence ending with parity 1
        max_alternating = 0
        for parity in parities:
            if parity == 0:
                dp0 = dp1 + 1
            else:
                dp1 = dp0 + 1
            max_alternating = max(max_alternating, dp0, dp1)
        
        # The answer is the maximum of the two scenarios
        return max(max_same_parity, max_alternating)