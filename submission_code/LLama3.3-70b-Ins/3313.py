from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        def calculate_strength(subarrays):
            strength = 0
            for i, subarray in enumerate(subarrays):
                subarray_sum = prefix_sum[subarray[1] + 1] - prefix_sum[subarray[0]]
                strength += ((-1) ** i) * subarray_sum * (k - i)
            return strength

        max_strength = float('-inf')
        def backtrack(start, subarrays):
            nonlocal max_strength
            if len(subarrays) == k:
                max_strength = max(max_strength, calculate_strength(subarrays))
                return
            for i in range(start, n):
                for j in range(i, n):
                    subarrays.append((i, j))
                    backtrack(j + 1, subarrays)
                    subarrays.pop()

        backtrack(0, [])
        return max_strength