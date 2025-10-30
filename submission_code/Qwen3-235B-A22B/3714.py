class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        
        # Precompute inverses up to 70
        inv = [0] * 71
        for i in range(1, 71):
            inv[i] = pow(i, MOD - 2, MOD)
        
        sum_min = 0
        sum_max = 0
        
        for i in range(n):
            a = nums[i]
            
            # Calculate contribution to sum_min (as minimum)
            m = n - i - 1
            t_max = min(k - 1, m)
            c = 1  # Starts with C(m, 0)
            s_min = c
            for t in range(1, t_max + 1):
                c = c * (m - t + 1) % MOD
                c = c * inv[t] % MOD
                s_min = (s_min + c) % MOD
            sum_min = (sum_min + a * s_min) % MOD
            
            # Calculate contribution to sum_max (as maximum)
            m2 = i
            t_max2 = min(k - 1, m2)
            c = 1  # Starts with C(m2, 0)
            s_max = c
            for t in range(1, t_max2 + 1):
                c = c * (m2 - t + 1) % MOD
                c = c * inv[t] % MOD
                s_max = (s_max + c) % MOD
            sum_max = (sum_max + a * s_max) % MOD
        
        return (sum_min + sum_max) % MOD