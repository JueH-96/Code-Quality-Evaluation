class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Sort the array
        nums.sort()
        
        # Precompute powers of 2 and their inverses modulo MOD
        pow2 = [1] * (n+1)   # pow2[i] = 2^i (mod MOD)
        for i in range(1, n+1):
            pow2[i] = (pow2[i-1] * 2) % MOD
        
        inv2_of_2 = pow(pow2[1], MOD-2, MOD)  # Modular inverse of 2
        inv2 = [1] * (n+1)   # inv2[i] = (2^i)^(-1) (mod MOD)
        for i in range(1, n+1):
            inv2[i] = (inv2[i-1] * inv2_of_2) % MOD
        
        # p[i] will store the sum of nums[k] * 2^(-k) for k in [0..i], all mod MOD
        p = [0] * n
        p[0] = (nums[0] * inv2[0]) % MOD  # inv2[0] = 1
        for i in range(1, n):
            p[i] = (p[i-1] + nums[i] * inv2[i]) % MOD
        
        # Calculate the final sum of powers
        total = 0
        
        # Add contribution of single-element subsets: a_j^3
        for j in range(n):
            val = (nums[j] % MOD)
            total = (total + (val * val % MOD) * val % MOD) % MOD
        
        # Add contribution for subsets of size >= 2
        # For each j, add a_j^2 * 2^(j-1) * (sum over i=0..j-1 of a_i * 2^(-i))
        for j in range(1, n):
            a_j_sq = (nums[j] * nums[j]) % MOD
            contrib = a_j_sq * pow2[j-1] % MOD
            contrib = contrib * p[j-1] % MOD
            total = (total + contrib) % MOD
        
        return total % MOD