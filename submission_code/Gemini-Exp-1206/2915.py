from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        count = 0
        prefix_counts = {0: 1}
        curr_count = 0
        for num in nums:
            if num % modulo == k:
                curr_count += 1
            remainder = curr_count % modulo
            target = (remainder - k) % modulo
            if target in prefix_counts:
                count += prefix_counts[target]
            prefix_counts[remainder] = prefix_counts.get(remainder, 0) + 1
        return count