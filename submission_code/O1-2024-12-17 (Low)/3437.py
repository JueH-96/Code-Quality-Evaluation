class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        import bisect
        
        # Edge case: if there's only one spell, that's our answer
        if len(power) == 1:
            return power[0]
        
        # Count how many times each damage value appears
        from collections import Counter
        count_map = Counter(power)
        
        # Get the unique damage values in sorted order
        unique_damage = sorted(count_map.keys())
        
        # f[i] will store "damage_value * frequency"
        # i.e. total damage if we decide to use all spells of damage_value
        f = [d * count_map[d] for d in unique_damage]
        
        # dp[i] = max total damage considering up to unique_damage[i]
        dp = [0] * len(unique_damage)
        
        # Base case: either pick the first damage value
        dp[0] = f[0]
        
        for i in range(1, len(unique_damage)):
            # Option 1: skip this damage value
            res = dp[i - 1]
            
            # Option 2: pick this damage value + dp of a non-conflicting index
            # We need the largest j < i such that unique_damage[i] - unique_damage[j] > 2
            damage_needed = unique_damage[i] - 2
            # j is the rightmost index where unique_damage[j] <= damage_needed - 1
            # but effectively we can do a bisect_left on damage_needed
            j = bisect.bisect_left(unique_damage, damage_needed) - 1
            
            # If j is valid, add dp[j]; else, just f[i]
            if j >= 0:
                res = max(res, dp[j] + f[i])
            else:
                res = max(res, f[i])
            
            dp[i] = res
        
        return dp[-1]