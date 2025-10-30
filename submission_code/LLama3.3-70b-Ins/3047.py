from typing import List
import math

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def is_perfect_square(n: int) -> bool:
            """Check if a number is a perfect square."""
            sqrt = math.sqrt(n)
            return sqrt == int(sqrt)

        max_sum = 0
        n = len(nums)

        # Generate all possible subsets
        for mask in range(1, 1 << n):
            subset_sum = 0
            is_complete = True

            # Check if the subset is complete
            for i in range(n):
                if (mask & (1 << i)) != 0:
                    subset_sum += nums[i]
                    for j in range(i + 1, n):
                        if (mask & (1 << j)) != 0:
                            if not is_perfect_square(nums[i] * nums[j]):
                                is_complete = False
                                break
                if not is_complete:
                    break

            # Update the maximum sum if the subset is complete
            if is_complete:
                max_sum = max(max_sum, subset_sum)

        return max_sum