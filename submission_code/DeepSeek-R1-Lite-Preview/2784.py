class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        nums.sort()
        
        # Precompute pow2[j] = 2^j % MOD
        pow2 = [1] * n
        for j in range(1, n):
            pow2[j] = (pow2[j-1] * 2) % MOD
        
        # Precompute inv2_power[j] = inv2^j % MOD
        inv2 = pow(2, MOD-2, MOD)
        inv2_power = [1] * n
        for j in range(1, n):
            inv2_power[j] = (inv2_power[j-1] * inv2) % MOD
        
        P_prev = 0
        sum_total = 0
        for j in range(n):
            if j >= 1:
                S_j = (pow2[j-1] * P_prev + nums[j]) % MOD
            else:
                S_j = nums[j]
            term = (nums[j] * nums[j] % MOD) * S_j % MOD
            sum_total = (sum_total + term) % MOD
            P_curr = (P_prev + nums[j] * inv2_power[j]) % MOD
            P_prev = P_curr
        return sum_total