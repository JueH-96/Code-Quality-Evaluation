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
        dp = [0] * (n + 1)
        
        for i in range(n-1, -1, -1):
            current_power = unique_powers[i]
            total = current_power * freq[current_power]
            
            # Find the next possible power that can be used
            next_index = i + 1
            while next_index < n and unique_powers[next_index] <= current_power + 2:
                next_index += 1
            
            # If next_index is within bounds, add dp[next_index] to total
            if next_index < n:
                total += dp[next_index]
            
            # Choose the maximum between taking current_power or not
            dp[i] = max(total, dp[i+1])
        
        return dp[0]