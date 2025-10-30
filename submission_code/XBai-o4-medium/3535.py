class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        
        # Initialize DP for the first element
        prev_dp = [1] * (nums[0] + 1)
        
        for i in range(1, n):
            current_num = nums[i]
            prev_num = nums[i-1]
            delta = max(0, current_num - prev_num)
            
            m = len(prev_dp)
            # Compute prefix sums of prev_dp
            prefix = [0] * (m + 1)
            for j in range(m):
                prefix[j+1] = (prefix[j] + prev_dp[j]) % MOD
            
            # Compute new_dp for current_num
            new_dp = [0] * (current_num + 1)
            for curr in range(0, current_num + 1):
                upper_prev = curr - delta
                if upper_prev < 0:
                    continue
                upper_prev_clamped = min(upper_prev, prev_num)
                new_dp[curr] = prefix[upper_prev_clamped + 1] % MOD
            
            prev_dp = new_dp
        
        return sum(prev_dp) % MOD