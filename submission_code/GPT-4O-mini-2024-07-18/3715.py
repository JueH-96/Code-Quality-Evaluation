from typing import List
from collections import defaultdict

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Create a dictionary to store the number of coins at each position
        coin_map = defaultdict(int)
        
        # Fill the coin_map with the number of coins in each bag
        for l, r, c in coins:
            for i in range(l, r + 1):
                coin_map[i] += c
        
        # Create a sorted list of unique positions where coins are present
        positions = sorted(coin_map.keys())
        
        # Initialize variables to find the maximum coins in k consecutive bags
        max_coins = 0
        current_coins = 0
        left = 0
        
        # Use a sliding window approach to find the maximum coins in k consecutive bags
        for right in range(len(positions)):
            current_coins += coin_map[positions[right]]
            
            # If we have more than k bags, slide the window
            while positions[right] - positions[left] + 1 > k:
                current_coins -= coin_map[positions[left]]
                left += 1
            
            # Update max_coins if current_coins is greater
            if positions[right] - positions[left] + 1 == k:
                max_coins = max(max_coins, current_coins)
        
        return max_coins