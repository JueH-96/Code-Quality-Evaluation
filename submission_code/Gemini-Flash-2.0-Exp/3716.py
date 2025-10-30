class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        dp = {}

        def solve(index, prev_index, prev_diff):
            if index == n:
                return 0

            if (index, prev_index, prev_diff) in dp:
                return dp[(index, prev_index, prev_diff)]

            ans = solve(index + 1, prev_index, prev_diff)  # Skip the current element

            if prev_index == -1:  # First element in the subsequence
                ans = max(ans, 1 + solve(index + 1, index, -1))
            else:
                curr_diff = abs(nums[index] - nums[prev_index])
                if prev_diff == -1 or curr_diff <= prev_diff:
                    ans = max(ans, 1 + solve(index + 1, index, curr_diff))

            dp[(index, prev_index, prev_diff)] = ans
            return ans

        return solve(0, -1, -1)