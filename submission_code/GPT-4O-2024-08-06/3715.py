class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Create a dictionary to store the number of coins at each position
        coin_map = {}
        
        # Fill the coin_map with the number of coins at each position
        for l, r, c in coins:
            for pos in range(l, r + 1):
                coin_map[pos] = c
        
        # Sort the positions to consider them in order
        sorted_positions = sorted(coin_map.keys())
        
        # Initialize variables for sliding window
        max_coins = 0
        current_coins = 0
        window_start = 0
        
        # Use a sliding window to find the maximum coins in k consecutive bags
        for window_end in range(len(sorted_positions)):
            pos = sorted_positions[window_end]
            current_coins += coin_map[pos]
            
            # If the window size exceeds k, slide the window
            if pos - sorted_positions[window_start] + 1 > k:
                current_coins -= coin_map[sorted_positions[window_start]]
                window_start += 1
            
            # Update the maximum coins found
            max_coins = max(max_coins, current_coins)
        
        return max_coins