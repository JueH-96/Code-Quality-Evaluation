class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [-1] * n
        dp[0] = 0
        
        q = [0]
        
        while q:
            curr = q.pop(0)
            
            for next_jump in range(curr + 1, n):
                if -target <= nums[next_jump] - nums[curr] <= target:
                    if dp[next_jump] == -1:
                        dp[next_jump] = dp[curr] + 1
                        q.append(next_jump)
                    
        return dp[n-1]