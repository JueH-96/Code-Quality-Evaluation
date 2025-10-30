from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Count homogeneous subsequences (all even or all odd)
        even_count = sum(1 for num in nums if num % 2 == 0)
        odd_count = len(nums) - even_count
        homogeneous = max(even_count, odd_count)
        
        # Greedy function to get maximum strictly alternating subsequence
        def max_alternating(start_parity: int) -> int:
            cnt = 0
            expected = start_parity
            for num in nums:
                if num % 2 == expected:
                    cnt += 1
                    # flip the expected parity (0 -> 1, 1 -> 0)
                    expected ^= 1
            return cnt if cnt >= 2 else 0  # valid subsequence must have at least 2
            
        # Try both possibilities: starting with even (0) or odd (1)
        alt1 = max_alternating(0)
        alt2 = max_alternating(1)
        alternating = max(alt1, alt2)
        
        return max(homogeneous, alternating)