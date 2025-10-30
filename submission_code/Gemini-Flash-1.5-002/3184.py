class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        dp = {}
        def solve(i, prev_index):
            if i == n:
                return 0
            if (i, prev_index) in dp:
                return dp[(i, prev_index)]
            
            # Option 1: Don't include nums[i]
            ans = solve(i + 1, prev_index)
            
            # Option 2: Include nums[i] if it satisfies the condition
            if prev_index == -1 or nums[i] - nums[prev_index] >= i - prev_index:
                ans = max(ans, nums[i] + solve(i + 1, i))
            
            dp[(i, prev_index)] = ans
            return ans
        
        return solve(0, -1)