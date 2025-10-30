from typing import List

class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        max_prod = -1

        def solve(index, current_sum, current_product, is_positive):
            nonlocal max_prod

            if index == len(nums):
                if current_sum == k:
                    max_prod = max(max_prod, current_product)
                return

            # Option 1: Exclude the current number
            solve(index + 1, current_sum, current_product, is_positive)

            # Option 2: Include the current number
            new_sum = current_sum + (nums[index] if is_positive else -nums[index])
            new_product = current_product * nums[index]

            if new_product <= limit:
                solve(index + 1, new_sum, new_product, not is_positive)

        solve(0, 0, 1, True)
        return max_prod