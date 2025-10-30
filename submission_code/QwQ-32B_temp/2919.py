class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        # Sort the usageLimits in non-decreasing order
        usageLimits.sort()
        # Reverse to get non-increasing order
        usageLimits.reverse()
        n = len(usageLimits)
        # Compute prefix sums
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + usageLimits[i - 1]
        max_k = 0
        # Check each possible k from 1 to n
        for k in range(1, n + 1):
            required = k * (k + 1) // 2
            if prefix[k] >= required:
                max_k = k
        return max_k