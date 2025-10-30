class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return n

        dp = {}  # Memoization to store results for subproblems

        def solve(index, prev_diff):
            if index == n:
                return 0

            if (index, prev_diff) in dp:
                return dp[(index, prev_diff)]

            max_len = solve(index + 1, prev_diff)  # Skip current element

            for i in range(index + 1, n):
                diff = abs(nums[i] - nums[index])
                if diff <= prev_diff:
                    max_len = max(max_len, 1 + solve(i + 1, diff))

            dp[(index, prev_diff)] = max_len
            return max_len

        max_length = 0
        for i in range(n):
            max_length = max(max_length, 1 + solve(i + 1, float('inf')))

        return max_length