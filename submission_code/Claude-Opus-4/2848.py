class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        
        # dp[mask][last] = number of ways to form a valid permutation
        # using elements represented by mask, ending with nums[last]
        dp = {}
        
        def solve(mask, last):
            if (mask, last) in dp:
                return dp[(mask, last)]
            
            # If all elements are used, we found a valid permutation
            if mask == (1 << n) - 1:
                return 1
            
            result = 0
            # Try adding each unused element
            for i in range(n):
                if mask & (1 << i):  # Already used
                    continue
                
                # Check if we can add nums[i] after nums[last]
                if nums[last] % nums[i] == 0 or nums[i] % nums[last] == 0:
                    result = (result + solve(mask | (1 << i), i)) % MOD
            
            dp[(mask, last)] = result
            return result
        
        # Start with each element as the first element
        total = 0
        for i in range(n):
            total = (total + solve(1 << i, i)) % MOD
        
        return total