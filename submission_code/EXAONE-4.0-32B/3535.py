MOD = 10**9 + 7

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * (nums[0] + 1)
        if n == 1:
            return sum(dp) % MOD
        
        for i in range(n - 1):
            curr_val = nums[i]
            next_val = nums[i + 1]
            d = next_val - curr_val
            threshold = max(0, d)
            
            M_curr = curr_val
            prefix = [0] * (M_curr + 1)
            if M_curr >= 0:
                prefix[0] = dp[0] % MOD
                for j in range(1, M_curr + 1):
                    prefix[j] = (prefix[j - 1] + dp[j]) % MOD
            
            new_dp = [0] * (next_val + 1)
            for y in range(next_val + 1):
                if y < threshold:
                    new_dp[y] = 0
                else:
                    x_max = min(M_curr, y - threshold)
                    new_dp[y] = prefix[x_max]
            
            dp = new_dp
        
        return sum(dp) % MOD