class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        from collections import Counter
        
        # Count frequency of each damage value
        damage_count = Counter(power)
        
        # Get sorted unique damage values
        unique_damages = sorted(damage_count.keys())
        n = len(unique_damages)
        
        if n == 0:
            return 0
        
        # dp[i] = maximum damage considering spells up to index i
        dp = [0] * n
        
        # Base case
        dp[0] = unique_damages[0] * damage_count[unique_damages[0]]
        
        for i in range(1, n):
            current_damage = unique_damages[i]
            current_total = current_damage * damage_count[current_damage]
            
            # Option 1: Don't take current damage value
            dp[i] = dp[i-1]
            
            # Option 2: Take current damage value
            # Find the last index j where unique_damages[j] < current_damage - 2
            j = i - 1
            while j >= 0 and unique_damages[j] >= current_damage - 2:
                j -= 1
            
            if j >= 0:
                dp[i] = max(dp[i], dp[j] + current_total)
            else:
                dp[i] = max(dp[i], current_total)
        
        return dp[n-1]