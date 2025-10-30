from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Count constant parity candidates
        even_count = 0
        odd_count = 0
        parities = []
        for x in nums:
            p = x & 1
            parities.append(p)
            if p == 0:
                even_count += 1
            else:
                odd_count += 1
        
        max_const = max(even_count, odd_count)
        
        # Compute longest alternating-parity subsequence for both possible starts
        def longest_alt(start_parity: int) -> int:
            need = start_parity
            length = 0
            for p in parities:
                if p == need:
                    length += 1
                    need ^= 1
            return length
        
        max_alt = max(longest_alt(0), longest_alt(1))
        
        return max(max_const, max_alt)