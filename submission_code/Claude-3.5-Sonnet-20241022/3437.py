class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        n = len(power)
        if n == 1:
            return power[0]
            
        # Group spells by damage value
        damage_count = {}
        for p in power:
            damage_count[p] = damage_count.get(p, 0) + 1
            
        # Get unique damage values
        unique_damages = sorted(damage_count.keys())
        m = len(unique_damages)
        
        # dp[i] represents max damage possible considering spells up to index i
        dp = [0] * m
        
        # Base case - first damage value
        dp[0] = unique_damages[0] * damage_count[unique_damages[0]]
        
        if m == 1:
            return dp[0]
            
        # Second damage value if it can be used with first
        if unique_damages[1] - unique_damages[0] > 2:
            dp[1] = dp[0] + unique_damages[1] * damage_count[unique_damages[1]]
        else:
            dp[1] = max(dp[0], unique_damages[1] * damage_count[unique_damages[1]])
            
        # Fill dp array
        for i in range(2, m):
            curr_damage = unique_damages[i]
            prev_damage = unique_damages[i-1]
            
            if curr_damage - prev_damage > 2:
                # Can use current with previous
                dp[i] = dp[i-1] + curr_damage * damage_count[curr_damage]
            else:
                # Check if using current alone is better than previous result
                use_current = curr_damage * damage_count[curr_damage]
                
                # Find last compatible damage value
                j = i - 1
                while j >= 0 and curr_damage - unique_damages[j] <= 2:
                    j -= 1
                    
                if j >= 0:
                    use_current += dp[j]
                    
                dp[i] = max(dp[i-1], use_current)
                
        return dp[m-1]