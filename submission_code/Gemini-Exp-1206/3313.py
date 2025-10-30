class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = {}

        def solve(index, count, sign):
            if count == k:
                return 0
            if index == n:
                return float('-inf')

            if (index, count, sign) in dp:
                return dp[(index, count, sign)]

            ans = solve(index + 1, count, sign)
            curr_sum = 0
            for i in range(index, n):
                curr_sum += nums[i]
                ans = max(ans, solve(i + 1, count + 1, -sign) + curr_sum * sign * (k - count))

            dp[(index, count, sign)] = ans
            return ans

        return solve(0, 0, 1)