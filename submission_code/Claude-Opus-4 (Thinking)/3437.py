class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        from collections import Counter
        import bisect
        
        # Count occurrences of each damage value
        damage_count = Counter(power)
        
        # Get sorted unique damage values
        unique_damages = sorted(damage_count.keys())
        n = len(unique_damages)
        
        if n == 0:
            return 0
        
        # dp[i] = maximum damage using damage values up to index i
        dp = [0] * n
        
        # Base case: first damage value
        dp[0] = unique_damages[0] * damage_count[unique_damages[0]]
        
        for i in range(1, n):
            current_damage = unique_damages[i]
            current_total = current_damage * damage_count[current_damage]
            
            # Option 1: Don't use current damage value
            dp[i] = dp[i-1]
            
            # Option 2: Use current damage value
            # Find the last damage value that we can include
            # (i.e., a value that is <= current_damage - 3)
            target = current_damage - 3
            j = bisect.bisect_right(unique_damages, target) - 1
            
            if j >= 0:
                dp[i] = max(dp[i], dp[j] + current_total)
            else:
                dp[i] = max(dp[i], current_total)
        
        return dp[n-1]