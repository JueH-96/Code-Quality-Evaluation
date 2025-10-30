class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Create a list to store the coins at each position
        coin_map = {}
        
        # Populate the coin_map with the coins from each segment
        for l, r, c in coins:
            for i in range(l, r + 1):
                coin_map[i] = c
        
        # Convert the coin_map to a sorted list of positions
        positions = sorted(coin_map.keys())
        
        # Create a prefix sum array for the coins
        prefix_sum = [0] * (len(positions) + 1)
        for i in range(len(positions)):
            prefix_sum[i + 1] = prefix_sum[i] + coin_map[positions[i]]
        
        # Find the maximum sum of k consecutive bags
        max_coins = 0
        for i in range(len(positions) - k + 1):
            max_coins = max(max_coins, prefix_sum[i + k] - prefix_sum[i])
        
        return max_coins