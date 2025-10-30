class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Sort the coins by their left index
        coins.sort(key=lambda x: x[0])
        
        # Initialize the total coins collected
        total_coins = 0
        
        # Iterate through the coins, collecting k consecutive bags
        for i in range(len(coins)):
            if i + k - 1 < len(coins):
                total_coins += sum(coins[i:i+k])
            else:
                break
        
        return total_coins