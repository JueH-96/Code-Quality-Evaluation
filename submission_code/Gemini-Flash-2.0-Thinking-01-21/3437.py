from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        unique_damages = sorted(list(set(power)))
        damage_counts = {}
        for p in power:
            damage_counts[p] = damage_counts.get(p, 0) + 1
        
        n = len(unique_damages)
        if n == 0:
            return 0
        dp = [0] * n
        
        for i in range(n):
            current_damage = unique_damages[i]
            count = damage_counts[current_damage]
            damage_if_include = current_damage * count
            
            prev_max_damage = 0
            prev_index = -1
            for j in range(i):
                if unique_damages[j] < current_damage - 2:
                    prev_index = j
                    
            if prev_index >= 0:
                prev_max_damage = dp[prev_index]
                
            damage_if_include += prev_max_damage
            damage_if_exclude = 0
            if i > 0:
                damage_if_exclude = dp[i-1]
            else:
                damage_if_exclude = 0
                
            dp[i] = max(damage_if_include, damage_if_exclude)
            
        return dp[n-1]