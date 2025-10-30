from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        dp = {}  # (index, num_subarrays) -> min_sum

        def solve(index, num_subarrays):
            if (index, num_subarrays) in dp:
                return dp[(index, num_subarrays)]

            if num_subarrays == 0:
                return 0 if index == -1 else float('inf')

            min_sum = float('inf')
            target_and = andValues[num_subarrays - 1]

            current_and = -1
            for start_index in range(index, -1, -1):
                if current_and == -1:
                    current_and = nums[start_index]
                else:
                    current_and &= nums[start_index]

                if current_and == target_and:
                    prev_sum = solve(start_index - 1, num_subarrays - 1)
                    if prev_sum != float('inf'):
                        min_sum = min(min_sum, prev_sum + nums[index])

            dp[(index, num_subarrays)] = min_sum
            return min_sum

        result = solve(n - 1, m)
        return result if result != float('inf') else -1