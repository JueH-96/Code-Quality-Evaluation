class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        
        # Precompute the running sum S[k] = sum_{i=0..k} nums[i]*2^(k-i)  (all modded).
        n = len(nums)
        S = [0]*n
        S[0] = nums[0] % MOD
        for i in range(1,n):
            S[i] = (2 * S[i-1] + nums[i]) % MOD
        
        ans = 0
        
        # 1) Add the contribution of all single-element subsets: arr[i]^3
        for x in nums:
            ans = (ans + (x % MOD) * (x % MOD) * (x % MOD)) % MOD
        
        # 2) Add the contribution of all subsets of size >= 2
        #    For each j from 1..n-1, the extra is arr[j]^2 * S[j-1]
        for j in range(1,n):
            sq = (nums[j] % MOD) * (nums[j] % MOD) % MOD
            ans = (ans + sq * S[j-1]) % MOD
        
        return ans