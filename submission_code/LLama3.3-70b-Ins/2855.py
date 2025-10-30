from typing import List

class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        memo = {}

        def dfs(i, jumps):
            if i == n - 1:
                return jumps
            if i in memo:
                return memo[i]
            max_jumps = -1
            for j in range(i + 1, n):
                if target >= nums[j] - nums[i] >= -target:
                    max_jumps = max(max_jumps, dfs(j, jumps + 1))
            memo[i] = max_jumps
            return max_jumps

        result = dfs(0, 0)
        return result if result != -1 else -1