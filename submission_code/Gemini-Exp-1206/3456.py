class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = {}

        def solve(index, prev, diff_count):
            if index == n:
                return 0
            
            state = (index, prev, diff_count)
            if state in dp:
                return dp[state]

            res = solve(index + 1, prev, diff_count)

            if diff_count + (nums[index] != prev) <= k or prev == -1:
                res = max(res, 1 + solve(index + 1, nums[index], diff_count + (nums[index] != prev)))

            dp[state] = res
            return res

        return solve(0, -1, 0)