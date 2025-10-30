from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        prefix_count = {0: 1}
        current_sum = 0
        count = 0

        for num in nums:
            if num % modulo == k:
                current_sum = (current_sum + 1) % modulo
            count += prefix_count.get(current_sum, 0)
            prefix_count[current_sum] = prefix_count.get(current_sum, 0) + 1

        return count