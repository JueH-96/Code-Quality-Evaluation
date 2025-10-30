class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        # Find positions where s1 and s2 differ
        diffs = [i for i in range(len(s1)) if s1[i] != s2[i]]
        
        # If odd number of differing positions, return -1
        if len(diffs) % 2 != 0:
            return -1
        
        n = len(diffs)
        # Edge case: no differences
        if n == 0:
            return 0
            
        # Memoization cache
        memo = {}
        
        def dp(mask):
            # Base case: no more positions to fix
            if mask == 0:
                return 0
                
            # Check if already computed
            if mask in memo:
                return memo[mask]
            
            # Find the first active bit
            i = 0
            while i < n and (mask & (1 << i)) == 0:
                i += 1
            
            result = float('inf')
            # Try pairing the first active position with each other active position
            for j in range(i+1, n):
                if mask & (1 << j):
                    # Create new mask without positions i and j
                    new_mask = mask & ~(1 << i) & ~(1 << j)
                    # Cost to pair these positions: min of using operation 1 or operation 2
                    cost = min(x, diffs[j] - diffs[i])
                    result = min(result, dp(new_mask) + cost)
            
            memo[mask] = result
            return result
        
        # Start with all positions active (all bits set)
        return dp((1 << n) - 1)