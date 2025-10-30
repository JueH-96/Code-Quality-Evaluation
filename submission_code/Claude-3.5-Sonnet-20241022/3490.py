class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = {}
        
        def solve(idx, prev, parity):
            if idx >= n:
                return 0
                
            state = (idx, prev, parity)
            if state in dp:
                return dp[state]
            
            # Don't take current element
            ans = solve(idx + 1, prev, parity)
            
            # Take current element if valid
            if prev == -1 or (prev + nums[idx]) % 2 == parity:
                take = 1 + solve(idx + 1, nums[idx], (prev + nums[idx]) % 2 if prev != -1 else -1)
                ans = max(ans, take)
                
            dp[state] = ans
            return ans
            
        # Try both parities as starting point
        result = max(solve(0, -1, 0), solve(0, -1, 1))
        
        return result