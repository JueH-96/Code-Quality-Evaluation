from bisect import bisect_right

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        if not power:
            return 0
        
        power.sort()
        
        # Create groups of consecutive elements
        groups = []
        n = len(power)
        i = 0
        while i < n:
            current = power[i]
            count = 1
            while i + count < n and power[i + count] == current:
                count += 1
            groups.append((current, count))
            i += count
        
        # Extract group values and their total damage
        group_values = [g[0] for g in groups]
        total_damage = [g[0] * g[1] for g in groups]
        m = len(groups)
        
        if m == 0:
            return 0
        
        # Initialize DP array
        dp = [0] * m
        dp[0] = total_damage[0]
        
        for i in range(1, m):
            target = group_values[i] - 3
            # Find the largest index j where group_values[j] <= target
            idx = bisect_right(group_values, target, 0, i) - 1
            if idx >= 0:
                option = total_damage[i] + dp[idx]
            else:
                option = total_damage[i]
            dp[i] = max(dp[i-1], option)
        
        return dp[-1]