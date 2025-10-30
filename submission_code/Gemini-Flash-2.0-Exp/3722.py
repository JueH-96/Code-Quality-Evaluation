class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        dp = {}

        def solve(idx, num_subarrays):
            if (idx, num_subarrays) in dp:
                return dp[(idx, num_subarrays)]

            if num_subarrays == 0:
                return 0

            if idx >= n:
                return float('-inf')

            ans = float('-inf')
            ans = max(ans, solve(idx + 1, num_subarrays))

            curr_sum = 0
            for j in range(idx, n):
                curr_sum += nums[j]
                if j - idx + 1 >= m:
                    ans = max(ans, curr_sum + solve(j + 1, num_subarrays - 1))

            dp[(idx, num_subarrays)] = ans
            return ans

        return solve(0, k)