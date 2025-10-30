class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        dp = {}

        def solve(index, prev_diff):
            if index == n:
                return 0

            if (index, prev_diff) in dp:
                return dp[(index, prev_diff)]
            
            res = solve(index + 1, prev_diff)  # Skip current element

            if prev_diff == -1 or abs(nums[index] - nums[index - 1]) <= prev_diff:
                res = max(res, 1 + solve(index + 1, abs(nums[index] - nums[index-1])))

            dp[(index, prev_diff)] = res
            return res

        ans = 0
        for i in range(n):
            ans = max(ans, 1 + solve(i + 1, abs(nums[i] - nums[i-1]) if i > 0 else -1))
            
        return ans