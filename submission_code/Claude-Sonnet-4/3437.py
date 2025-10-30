class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        from collections import Counter
        
        # Count frequency of each damage value
        freq = Counter(power)
        
        # Get sorted unique damage values
        unique_powers = sorted(freq.keys())
        n = len(unique_powers)
        
        if n == 1:
            return unique_powers[0] * freq[unique_powers[0]]
        
        # dp[i] represents max damage we can get considering first i unique powers
        dp = [0] * n
        
        # Base case: first power
        dp[0] = unique_powers[0] * freq[unique_powers[0]]
        
        # For second power, we need to check if we can take both first and second
        if unique_powers[1] - unique_powers[0] <= 2:
            # Can't take both, take the better one
            dp[1] = max(dp[0], unique_powers[1] * freq[unique_powers[1]])
        else:
            # Can take both
            dp[1] = dp[0] + unique_powers[1] * freq[unique_powers[1]]
        
        # Fill the dp array
        for i in range(2, n):
            current_damage = unique_powers[i] * freq[unique_powers[i]]
            
            # Option 1: Don't take current power
            dp[i] = dp[i-1]
            
            # Option 2: Take current power
            # Find the latest power we can take along with current power
            j = i - 1
            while j >= 0 and unique_powers[i] - unique_powers[j] <= 2:
                j -= 1
            
            if j >= 0:
                dp[i] = max(dp[i], dp[j] + current_damage)
            else:
                dp[i] = max(dp[i], current_damage)
        
        return dp[n-1]