class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Modular inverse of 2
        inv2 = pow(2, MOD - 2, MOD)
        
        # f[j] = sum of 2^(n - length(T)) over all subsequences T that sum to j
        f = [0] * (k + 1)
        f[0] = pow(2, n, MOD)  # The empty subsequence
        
        for i in range(n):
            new_f = f[:]
            for j in range(nums[i], k + 1):
                new_f[j] = (new_f[j] + f[j - nums[i]] * inv2) % MOD
            f = new_f
        
        return f[k]