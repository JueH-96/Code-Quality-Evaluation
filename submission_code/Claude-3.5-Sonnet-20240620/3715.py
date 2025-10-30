class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Sort coins based on the left coordinate
        coins.sort(key=lambda x: x[0])
        
        # Create a list of all positions with their coin values
        positions = []
        for l, r, c in coins:
            positions.append((l, c))
            positions.append((r + 1, -c))
        
        # Sort positions
        positions.sort()
        
        # Initialize variables
        current_coins = 0
        max_coins = 0
        window_sum = 0
        left = 0
        
        # Iterate through positions
        for right, coins_change in positions:
            # Update current coins
            current_coins += coins_change
            
            # Expand window
            while left < len(positions) and right - positions[left][0] + 1 > k:
                window_sum -= current_coins * (positions[left + 1][0] - positions[left][0])
                left += 1
            
            # Update window sum
            if right > positions[left][0]:
                window_sum += current_coins * (right - max(right - k, positions[left][0]) + 1)
            
            # Update max_coins
            max_coins = max(max_coins, window_sum)
        
        return max_coins