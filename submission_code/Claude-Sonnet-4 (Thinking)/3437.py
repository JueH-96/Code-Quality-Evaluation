class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        from collections import Counter
        import bisect
        
        # Count frequency of each damage value
        count = Counter(power)
        damages = sorted(count.keys())
        n = len(damages)
        
        if n == 0:
            return 0
        
        # dp[i] = maximum damage considering damages[0] to damages[i]
        dp = [0] * n
        
        for i in range(n):
            # Option 1: don't take damages[i]
            prev_dp = dp[i-1] if i > 0 else 0
            
            # Option 2: take all spells of damages[i]
            current_value = damages[i] * count[damages[i]]
            
            # Find the largest index j such that we can take damages[j] with damages[i]
            # This means damages[j] < damages[i] - 2
            j = bisect.bisect_left(damages, damages[i] - 2) - 1
            
            take_value = current_value + (dp[j] if j >= 0 else 0)
            
            dp[i] = max(prev_dp, take_value)
        
        return dp[n-1]