class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        
        # Initialize DP
        dp = []
        max_v = nums[0]
        dp0 = [1] * (max_v + 1)
        dp.append(dp0)
        
        for i in range(1, n):
            current_num = nums[i]
            prev_dp = dp[i-1]
            current_max_v = current_num
            dp_i = [0] * (current_max_v + 1)
            
            for v_prev in range(len(prev_dp)):
                if prev_dp[v_prev] == 0:
                    continue
                delta = current_num - nums[i-1]
                if delta > 0:
                    min_v = v_prev + delta
                else:
                    min_v = v_prev
                if min_v > current_max_v:
                    continue
                # Update all valid v_curr
                for v_curr in range(min_v, current_max_v + 1):
                    if v_curr < len(dp_i):
                        dp_i[v_curr] = (dp_i[v_curr] + prev_dp[v_prev]) % MOD
            
            dp.append(dp_i)
        
        if n == 0:
            return 0
        else:
            return sum(dp[-1]) % MOD