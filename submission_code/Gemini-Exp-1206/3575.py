class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = {}

        def solve(idx, count, first_half, second_half):
            if count == 2 * k:
                return first_half ^ second_half
            if idx == n:
                return -1

            state = (idx, count, first_half, second_half)
            if state in dp:
                return dp[state]

            res = solve(idx + 1, count, first_half, second_half)

            if count < k:
                res = max(res, solve(idx + 1, count + 1, first_half | nums[idx], second_half))
            else:
                res = max(res, solve(idx + 1, count + 1, first_half, second_half | nums[idx]))

            dp[state] = res
            return res

        return solve(0, 0, 0, 0)