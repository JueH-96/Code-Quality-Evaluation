class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        
        # Q[j] will represent the sum_{i=0..j-1} of (nums[i] * 2^(j-1 - i)), 1 <= j <= n
        # We'll build Q via the recurrence: Q[j+1] = 2 * Q[j] + nums[j].
        
        n = len(nums)
        Q = [0]*(n+1)
        # Base case: Q[1] = nums[0]
        Q[1] = nums[0]
        for j in range(1, n):
            Q[j+1] = (2*Q[j] + nums[j]) % MOD
        
        # Sum of a[i]^3 for i = 0..n-1
        cube_sum = 0
        for x in nums:
            cube_sum = (cube_sum + x*x*x) % MOD
        
        # Sum over j=1..n-1 of nums[j]^2 * Q[j]
        pair_sum = 0
        for j in range(1, n):
            pair_sum = (pair_sum + (nums[j]*nums[j])%MOD * Q[j]) % MOD
        
        return (cube_sum + pair_sum) % MOD