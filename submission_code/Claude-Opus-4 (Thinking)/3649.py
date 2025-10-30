class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        dp = [float('inf')] * (1 << n)
        dp[0] = 0
        
        for mask in range(1 << n):
            if dp[mask] == float('inf'):
                continue
            
            # Count how many locks have been broken
            broken_count = bin(mask).count('1')
            X = 1 + broken_count * K
            
            # Try breaking each unbroken lock
            for i in range(n):
                if mask & (1 << i):  # Lock i is already broken
                    continue
                
                # Calculate time to break lock i
                time_to_break = (strength[i] + X - 1) // X  # Ceiling division
                new_mask = mask | (1 << i)
                dp[new_mask] = min(dp[new_mask], dp[mask] + time_to_break)
        
        return dp[(1 << n) - 1]