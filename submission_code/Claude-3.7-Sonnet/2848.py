class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        
        # Use memoization to avoid recalculating states
        dp = {}
        
        def dfs(mask, last_idx):
            # If all elements are used, we've found a valid permutation
            if mask == (1 << n) - 1:
                return 1
            
            # Check if we've already computed this state
            if (mask, last_idx) in dp:
                return dp[(mask, last_idx)]
            
            result = 0
            for i in range(n):
                # Skip if element i is already used
                if mask & (1 << i):
                    continue
                
                # For the first element (last_idx == -1), we can choose any element
                # Otherwise, check the divisibility condition
                if last_idx == -1 or nums[last_idx] % nums[i] == 0 or nums[i] % nums[last_idx] == 0:
                    result = (result + dfs(mask | (1 << i), i)) % MOD
            
            dp[(mask, last_idx)] = result
            return result
        
        # Start with an empty permutation (no elements used yet)
        return dfs(0, -1)