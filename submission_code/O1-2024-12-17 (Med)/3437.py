class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        from bisect import bisect_left
        
        # Frequency map: damage_value -> count of spells
        freq = {}
        for dmg in power:
            freq[dmg] = freq.get(dmg, 0) + 1
        
        # Create a list of (distinct_damage_value, total_damage_for_that_value)
        # total_damage_for_that_value = damage_value * count_of_that_damage
        distinct_vals = sorted(freq.keys())
        total_for_val = [d * freq[d] for d in distinct_vals]
        
        # DP array: dp[i] will store the maximum sum we can achieve
        # considering damage values up to index i (in the sorted distinct list).
        dp = [0] * len(distinct_vals)
        
        # Base case: if we pick the first distinct damage value
        dp[0] = total_for_val[0]
        
        for i in range(1, len(distinct_vals)):
            # We'll find the largest j < i such that distinct_vals[i] - distinct_vals[j] >= 3
            # which means they do not conflict (since picking x forbids x-1, x-2, x+1, x+2).
            dmg_i = distinct_vals[i]
            # We want v[j] <= dmg_i - 3
            j = bisect_left(distinct_vals, dmg_i - 2) - 1
            
            # If we pick the i-th value:
            if j >= 0:
                pick_val = dp[j] + total_for_val[i]
            else:
                pick_val = total_for_val[i]
            
            # dp[i] is the maximum of either skipping this value (dp[i-1]) or picking it
            dp[i] = max(dp[i-1], pick_val)
        
        return dp[-1]