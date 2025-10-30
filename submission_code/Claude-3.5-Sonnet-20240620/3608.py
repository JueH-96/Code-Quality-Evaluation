class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        n = len(nums)
        dp = [0] * (1 << n)
        dp[0] = 1
        
        gcd_count = {}
        
        for mask in range(1, 1 << n):
            current_gcd = 0
            for i in range(n):
                if mask & (1 << i):
                    current_gcd = gcd(current_gcd, nums[i])
            dp[mask] = pow(2, bin(mask).count('1') - 1, MOD)
            gcd_count[current_gcd] = gcd_count.get(current_gcd, 0) + dp[mask]
        
        result = 0
        for count in gcd_count.values():
            result = (result + count * count) % MOD
        
        return (result - pow(2, n, MOD) + MOD) % MOD