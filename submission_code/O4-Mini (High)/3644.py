from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        # Build prefix sum array
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]

        best = float('inf')
        # Try every starting index i
        for i in range(n):
            # j is the exclusive end index; subarray length = j - i
            # length between l and r => j in [i+l, i+r], but j <= n
            for j in range(i + l, min(n, i + r) + 1):
                s = prefix[j] - prefix[i]
                if s > 0 and s < best:
                    best = s

        return best if best != float('inf') else -1