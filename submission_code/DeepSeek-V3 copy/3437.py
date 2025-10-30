from collections import defaultdict

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        # Count the frequency of each power
        freq = defaultdict(int)
        for p in power:
            freq[p] += 1
        
        # Get sorted unique powers
        unique_powers = sorted(freq.keys())
        n = len(unique_powers)
        
        # Initialize dp array
        dp = [0] * n
        
        for i in range(n):
            current_power = unique_powers[i]
            total = current_power * freq[current_power]
            # Find the last power that can be included
            j = i - 1
            while j >= 0 and unique_powers[j] >= current_power - 2:
                j -= 1
            if j >= 0:
                total += dp[j]
            # Compare with the previous dp value
            if i > 0:
                dp[i] = max(dp[i-1], total)
            else:
                dp[i] = total
        
        return dp[-1]