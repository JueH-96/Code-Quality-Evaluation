class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        # Create a set of the available coins
        coin_set = set(coins)
        
        # Initialize a list to store the minimum number of coins needed for each value from 1 to target
        dp = [float('inf')] * (target + 1)
        dp[0] = 0
        
        # Iterate through the values from 1 to target
        for i in range(1, target + 1):
            # Check if the current value is already in the coin set
            if i in coin_set:
                dp[i] = 1
            else:
                # Try subtracting each available coin from the current value
                for coin in coins:
                    if i - coin >= 0:
                        dp[i] = min(dp[i], dp[i - coin] + 1)
        
        # If the target value is not obtainable, return -1
        if dp[target] == float('inf'):
            return -1
        else:
            return dp[target]